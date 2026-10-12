from pathlib import Path
import json
def write_json(path, value):
    def clean(v):
        if hasattr(v,'tolist'):return clean(v.tolist())
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,float) and not __import__('math').isfinite(v): return None
        return v
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, mesh, xs, us):
    import numpy as np
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,xs=np.array(xs),controls=np.concatenate(us),
                            offsets=np.r_[0,np.cumsum([len(u) for u in us])],mesh_json=json.dumps(mesh))
    temporary.replace(path)


