#!/usr/bin/env python3
import collections, importlib.util, operator, torch
from pathlib import Path
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

_probe_path=Path(__file__).with_name('flux2_lod_aot_probe.py')
spec=importlib.util.spec_from_file_location('probe', _probe_path)
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def _const_int(v, node):
    """Coerce a possibly-symbolic size expression to a plain int.

    Under a dynamic-shape capture the output_padding terms arrive as SymInts.
    For every stride-1 convolution the expression algebraically collapses to 0
    (base == input size), so it is constant even though its type is symbolic,
    and ONNX needs a literal for the ConvTranspose attribute.  Anything that is
    genuinely shape-dependent is refused loudly rather than silently guarded to
    whatever the tracing example happened to be.
    """
    if isinstance(v, int):
        return v
    # int(SymInt) would *guard* a shape-dependent expression, silently baking the
    # traced size into the graph and producing a model that is quietly wrong at
    # every other resolution.  Only accept expressions with no free symbols.
    expr = getattr(getattr(v, 'node', None), 'expr', None)
    if expr is not None and not expr.free_symbols:
        return int(expr)
    raise RuntimeError(
        f'output_padding for {node} is shape-dependent ({v}); it cannot be a '
        f'static ONNX attribute, and guarding it would bake the traced size. '
        f'Only stride-1 convolutions are dynamic-safe.')


def _val(n): return getattr(n,'meta',{}).get('val')
def _shape(n):
    v=_val(n); return tuple(v.shape) if hasattr(v,'shape') else None


def _replace_tuple0(g, node, new):
    users=list(node.users)
    g0=[]
    for u in users:
        if u.op=='call_function' and u.target==operator.getitem:
            idx=u.args[1]
            if idx==0: g0.append(u)
            elif idx in (1,2) and len(u.users)==0: pass
            else: raise RuntimeError(f'live unexpected tuple output {u}')
        else:
            raise RuntimeError(f'unexpected tuple user {u}')
    if len(g0)!=1: raise RuntimeError(f'expected one live output0, got {g0}')
    g0[0].replace_all_uses_with(new)
    for u in users:
        if len(u.users)==0: g.erase_node(u)
    g.erase_node(node)


def lower_conv_backward_input(gm):
    g=gm.graph
    nodes=[n for n in g.nodes if n.op=='call_function' and n.target==torch.ops.aten.convolution_backward.default]
    snap={n:(_shape(n.args[0]),_shape(n.args[1]),_shape(n.args[2])) for n in nodes}
    for cb in nodes:
        grad, inp, weight, bias_sizes, stride, padding, dilation, transposed, old_opad, groups, mask = cb.args
        if transposed is not False or list(mask)!=[True,False,False]:
            raise RuntimeError(f'unsupported conv backward form {transposed=} {mask=}')
        gsh,ish,wsh=snap[cb]
        s=list(stride); p=list(padding); d=list(dilation)
        if len(s)==1:s*=2
        if len(p)==1:p*=2
        if len(d)==1:d*=2
        kh,kw=wsh[-2:]
        if s[0]==1 and s[1]==1:
            # For stride 1 the algebra collapses exactly: the forward output is
            # gsh = ish + 2p - d*(k-1), so base == ish and output_padding is 0 at
            # every resolution.  Taking it analytically keeps the value a literal
            # under dynamic shapes, where the symbolic form is 's - s**2//s' and
            # sympy will not simplify integer floor division.
            opad=[0,0]
        else:
            base_h=(gsh[-2]-1)*s[0]-2*p[0]+d[0]*(kh-1)+1
            base_w=(gsh[-1]-1)*s[1]-2*p[1]+d[1]*(kw-1)+1
            opad=[_const_int(ish[-2]-base_h, cb), _const_int(ish[-1]-base_w, cb)]
        with g.inserting_before(cb):
            dx=g.call_function(torch.ops.aten.convolution.default,
                args=(grad,weight,None,s,p,d,True,opad,groups))
        _replace_tuple0(g,cb,dx)
    return len(nodes)


def lower_group_norm_backward_input(gm):
    g=gm.graph
    nodes=[n for n in g.nodes if n.op=='call_function' and n.target==torch.ops.aten.native_group_norm_backward.default]
    for bn in nodes:
        grad, inp, mean, rstd, weight, N, C, HxW, G, mask = bn.args
        if list(mask)!=[True,False,False]:
            raise RuntimeError(f'unsupported groupnorm backward mask {mask}')
        ish=_shape(inp)
        if not ish or len(ish)<3:
            raise RuntimeError(f'static rank>=3 [N,C,...] shape required, got {ish}')
        cpg=C//G
        # Under dynamic shapes HxW arrives as a graph Node, so M is not an int.
        # The formula does not actually need it: s1/M and s2/M are exactly the
        # means over the group axis, so using mean.dim instead of sum + divide
        # removes every dependence on M and works at any resolution.
        dynamic = not isinstance(HxW, int)
        M = None if dynamic else cpg*HxW
        # N, C, HxW and G arrive as explicit op args, so the formula itself is
        # rank-agnostic; only the affine broadcast depends on the input rank.
        # Diffusers' AttnProcessor normalizes [N,C,HW] (rank 3), the conv path
        # normalizes [N,C,H,W] (rank 4); both must lower.
        wshape=[1,C]+[1]*(len(ish)-2)
        with g.inserting_before(bn):
            if weight is None:                      # affine=False GroupNorm
                dyg4=grad
            else:
                w4=g.call_function(torch.ops.aten.view.default,args=(weight,wshape))
                dyg4=g.call_function(torch.ops.aten.mul.Tensor,args=(grad,w4))
            gshape=[N,G,-1] if dynamic else [N,G,M]
            dyg=g.call_function(torch.ops.aten.view.default,args=(dyg4,gshape))
            xg=g.call_function(torch.ops.aten.view.default,args=(inp,gshape))
            mu=g.call_function(torch.ops.aten.unsqueeze.default,args=(mean,-1))
            rs=g.call_function(torch.ops.aten.unsqueeze.default,args=(rstd,-1))
            xc=g.call_function(torch.ops.aten.sub.Tensor,args=(xg,mu))
            xhat=g.call_function(torch.ops.aten.mul.Tensor,args=(xc,rs))
            # dx = rstd * (dyg - mean(dyg) - xhat * mean(dyg*xhat))
            m1=g.call_function(torch.ops.aten.mean.dim,args=(dyg,[2],True))
            prod=g.call_function(torch.ops.aten.mul.Tensor,args=(dyg,xhat))
            m2=g.call_function(torch.ops.aten.mean.dim,args=(prod,[2],True))
            t1=g.call_function(torch.ops.aten.sub.Tensor,args=(dyg,m1))
            xhm2=g.call_function(torch.ops.aten.mul.Tensor,args=(xhat,m2))
            t2=g.call_function(torch.ops.aten.sub.Tensor,args=(t1,xhm2))
            dxg=g.call_function(torch.ops.aten.mul.Tensor,args=(rs,t2))
            dx=g.call_function(torch.ops.aten.view.default,args=(dxg,list(ish)))
        _replace_tuple0(g,bn,dx)
    return len(nodes)


def lower_softmax_backward(gm):
    g=gm.graph
    nodes=[n for n in g.nodes if n.op=='call_function' and n.target==torch.ops.aten._softmax_backward_data.default]
    for n in nodes:
        grad,out,dim,dtype=n.args
        with g.inserting_before(n):
            go=g.call_function(torch.ops.aten.mul.Tensor,args=(grad,out))
            s=g.call_function(torch.ops.aten.sum.dim_IntList,args=(go,[dim],True))
            centered=g.call_function(torch.ops.aten.sub.Tensor,args=(grad,s))
            dx=g.call_function(torch.ops.aten.mul.Tensor,args=(centered,out))
        n.replace_all_uses_with(dx); g.erase_node(n)
    return len(nodes)



def lower_sigmoid_backward(gm):
    g=gm.graph
    nodes=[n for n in g.nodes if n.op=='call_function' and n.target==torch.ops.aten.sigmoid_backward.default]
    for n in nodes:
        grad,out=n.args
        with g.inserting_before(n):
            neg_out=g.call_function(torch.ops.aten.neg.default,args=(out,))
            one_minus=g.call_function(torch.ops.aten.add.Scalar,args=(neg_out,1.0))
            prod=g.call_function(torch.ops.aten.mul.Tensor,args=(out,one_minus))
            dx=g.call_function(torch.ops.aten.mul.Tensor,args=(grad,prod))
        n.replace_all_uses_with(dx); g.erase_node(n)
    return len(nodes)

def lower_sign(gm):
    g=gm.graph
    targets={torch.ops.aten.sgn.default, torch.ops.aten.sign.default}
    nodes=[n for n in g.nodes if n.op=='call_function' and n.target in targets]
    for n in nodes:
        x=n.args[0]
        with g.inserting_before(n):
            pos=g.call_function(torch.ops.aten.gt.Scalar,args=(x,0.0))
            neg=g.call_function(torch.ops.aten.lt.Scalar,args=(x,0.0))
            posf=g.call_function(torch.ops.aten._to_copy.default,args=(pos,),kwargs={'dtype':torch.float32})
            negf=g.call_function(torch.ops.aten._to_copy.default,args=(neg,),kwargs={'dtype':torch.float32})
            y=g.call_function(torch.ops.aten.sub.Tensor,args=(posf,negf))
        n.replace_all_uses_with(y); g.erase_node(n)
    return len(nodes)


def strip_detach(gm):
    g=gm.graph; k=0
    for n in list(g.nodes):
        if n.op=='call_function' and n.target==torch.ops.aten.detach.default:
            n.replace_all_uses_with(n.args[0]); g.erase_node(n); k+=1
    return k



def normalize_layout_ops(gm):
    """Normalize view/expand/transpose ops to variants registered by WebGPU."""
    g=gm.graph
    stats={'view_copy':0,'expand_copy':0,'permute':0}
    for n in list(g.nodes):
        if n.op!='call_function':
            continue
        if n.target in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
            with g.inserting_before(n):
                y=g.call_function(torch.ops.aten.view_copy.default,args=n.args,kwargs=n.kwargs)
            n.replace_all_uses_with(y); g.erase_node(n); stats['view_copy']+=1
        elif n.target==torch.ops.aten.expand.default:
            with g.inserting_before(n):
                y=g.call_function(torch.ops.aten.expand_copy.default,args=n.args,kwargs=n.kwargs)
            n.replace_all_uses_with(y); g.erase_node(n); stats['expand_copy']+=1
        elif n.target==torch.ops.aten.t.default:
            sh=_shape(n)
            if sh is None or len(sh)!=2: raise RuntimeError('aten.t expected static rank-2')
            with g.inserting_before(n):
                y=g.call_function(torch.ops.aten.permute.default,args=(n.args[0],[1,0]))
            n.replace_all_uses_with(y); g.erase_node(n); stats['permute']+=1
        elif n.target==torch.ops.aten.transpose.int:
            sh=_shape(n)
            if sh is None: raise RuntimeError('transpose requires static rank metadata')
            rank=len(sh); d0=int(n.args[1]); d1=int(n.args[2])
            if d0<0:d0+=rank
            if d1<0:d1+=rank
            dims=list(range(rank)); dims[d0],dims[d1]=dims[d1],dims[d0]
            with g.inserting_before(n):
                y=g.call_function(torch.ops.aten.permute.default,args=(n.args[0],dims))
            n.replace_all_uses_with(y); g.erase_node(n); stats['permute']+=1
    return stats

def rewrite(gm, example_args=None):
    stats={}
    stats['conv_bwd']=lower_conv_backward_input(gm)
    stats['gn_bwd']=lower_group_norm_backward_input(gm)
    stats['softmax_bwd']=lower_softmax_backward(gm)
    stats['sigmoid_bwd']=lower_sigmoid_backward(gm)
    stats['sign']=lower_sign(gm)
    stats['detach']=strip_detach(gm)
    stats['layout']=normalize_layout_ops(gm)
    gm.graph.eliminate_dead_code(); gm.graph.lint(); gm.recompile()
    if example_args is not None:
        # Nodes created by the passes above carry empty meta.  Export paths and
        # _shape() both require meta['val'], so re-propagate over the whole graph.
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        from torch._subclasses.fake_tensor import FakeTensorMode as _FTM
        mode=next((a.fake_mode for a in example_args
                   if hasattr(a,'fake_mode') and a.fake_mode is not None), None)
        FakeTensorProp(gm, mode=mode or _FTM(allow_non_fake_inputs=True)).propagate(*example_args)
    return stats


def opname(n):
    if n.op!='call_function': return None
    t=n.target
    if hasattr(t,'_schema'):
        return str(t._schema.name)+(('.'+t._schema.overload_name) if t._schema.overload_name else '')
    return str(t)

if __name__=='__main__':
    m=probe.JointLOD('sym4').eval(); mode=FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        z=torch.empty(1,32,32,32,requires_grad=True); target=torch.empty(1,3,256,256)
        gm,sig=aot_export_module(m,(z,target),trace_joint=True,output_loss_index=0)
    stats=rewrite(gm)
    counts=collections.Counter(opname(n) for n in gm.graph.nodes if opname(n))
    print('rewrites',stats)
    print('nodes=',sum(counts.values()),'unique=',len(counts))
    for k,v in sorted(counts.items()): print(f'{v:4d}  {k}')
    print('remaining backward=',[(k,v) for k,v in counts.items() if 'backward' in k])
