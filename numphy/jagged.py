import numpy as np
from .util import ellipsis_eqiv, slice_len
from .core import Trace


class cachedProperty(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Uniform(object):
    _backend = np

    def __init__(self, _base, _missing=None, _prefix=(), _suffix=(), _links={}, **kwargs):
        _prefix = Trace(_prefix)
        self._base = base
        self._links = {k: prefix + v for k, v in _links.items()}
        self._missing = _missing and Trace(_missing)
        self._suffix = Trace(_suffix)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        if key in self._links:
            ret = self.__dict__[key] = (self._links[key] + self._suffix)(key, self._base)
            return ret
        else:
            return super(Uniform, self).__getattr__(self, key)

    def __setattr__(self, key, value):
        if key not in self._links and key not in self.__dict__ and self._missing is not None:
            self._links[key] = self._missing + key
        if key in self._links:
            value = (self._links[key] + self._suffix)(key, self._base, value)
        super(Uniform, self).__setattr__(key, value)

    def __getitem__(self, key, **kwargs):
        data = dict(
            (
                (k, v[key])
                for k, v in self.__dict__.items()
                if not (k.startswith("_") or k in self._links)
            ),
            _base=self._base,
            _links=self._links,
            _missing=self._missing,
            _suffix=self._suffix + key,
        )
        data.update(kwargs)
        return self.__class__(**data)

    @classmethod
    def of(cls, *others, **kwargs):
        cache = cls.__dict__.setdefault("_of_cache", {})
        value = cache.get(others, None)
        if value is None:
            name = kwargs.get("name", None) or sum((o.__name__ for o in others), cls.__name__)
            value = type(name, (cls,) + others, dict(kwargs.get("extras", {})))
            cache[others] = value
        return value

    def __exit__(self, *args):
        pass


class Jagged(Uniform):
    def __init__(self, _base, _sizes, **kwargs):
        super(JaggedData, self).__init__(_base=_base, **kwargs)
        if isinstance(_sizes, Trace):
            _sizes = _sizes(self._base)
        self._sizes = _sizes

    @cachedProperty
    def _total(self):
        return int(self._sizes.sum(axis=-1))

    @cachedProperty
    def _shape_flat(self):
        return (self._total,)

    @cachedProperty
    def _stops(self):
        return self._sizes.cumsum(axis=-1)

    @cachedProperty
    def _starts(self):
        return self._stops - self._sizes

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key)
        if len(key) == 0:  # identity
            return self
        elif len(key) == 1:  # compact view
            if self._backend.is_array(key):
                if key.dtype == bool:
                    if key.shape != self._shape_flat:
                        raise RuntimeError("mask expceted shape %s but got %s" % (
                            self._shape_flat,
                            key.shape,
                        ))
                    bkey = key
                elif key.dtype == int:
                    bkey = self._backend.zeros(self._shape_flat, dtype=bool)
                    bkey[key] = True
                    if 2 * key.size > bkey.size: # optimize if needed
                        key = bkey
                else:
                    raise RuntimeError("invalid index array type: %s" % key.dtype)
                return super(Jagged, self).__getitem__(
                    key,
                    _sizes=self._sum(bkey)
                )
            else:
                raise RuntimeError("invalid item access: %r" % key)
        else:  # expanded view
            # uniform and jagged key
            ukey, jkey = key[:-1], key[-1]
            # process uniform selection
            if ellipsis_eqiv(ukey):
                starts = self._starts
                stops = self._stops
                ukey = Ellipsis
            else:
                starts = self._starts[ukey]
                stops = self._stops[ukey]
                if starts[:-1] == stops[1:]: # can produce 1 large slice
                    return self.__getitem__(
                        slice(starts[0], stops[-1]),
                        _sizes=self._sizes[ukey]
                    )[..., jkey]
            if isinstance(jkey, int): # single item output (process via slice)
                return self[ukey, jkey:(None if jey == -1 else jkey + 1)][..., 0]
            if isinstance(jkey, slice): # predictable output (process via index-array)
                if slice_len(jkey) is None:
                    raise RuntimeError("jagged key %r has unpredictable length" % jkey)
                # TODO: optimize this
                jkey = np.stack([
                    range(sta, sto)[jkey]
                    for sta, sto in np.nditer([starts, stops])
                ], axis=0)
            if self._backend.is_array(jkey):
                if jkey.dtype == bool:
                    if jkey.shape != self._shape_flat:
                        raise RuntimeError("mask expceted shape %s but got %s" % (
                            self._shape_flat,
                            jkey.shape,
                        ))
                    return super(Jagged, self).__getitem__(
                        jkey,
                        _sizes=self._sum(jkey),
                    )
                elif jkey.dtype == int:
                    # apply ukey for same shape output
                    if jkey.shape[:-1] == self._sizes.shape[:-1]:
                        jkey = jkey[ukey, :]
                    elif ukey is not Ellipsis:
                        raise RuntimeError("invalid uniform access: %r" % ukey)
                    # TODO: transform to non-jagged
                    return super(Jagged, self).__getitem__(
                        jkey,
                        _sizes=np.full(jkey.shape[:-1], jkey.shape[-1], dtype=np.intp)
                    )
                else:
                    raise RuntimeError("invalid jagged access: %r" % jkey)
            if isinstance(jkey, JaggedArray):
                if jkey.dtype == bool:
                    pass # TODO
                elif jkey.dtype == int:
                    pass # TODO
                else:
                    raise RuntimeError("invalid jagged access: %r" % jkey)
            raise RuntimeError("invalid access: %r" % jkey)

    def _sum(self, data):
        csum = data.cumsum(axis=-1)
        csum = csum[self._sizes - 1]
        csum[1:] -= csum[:-1]
        return csum

    @classmethod
    def of(cls, *others, **kwargs):
        name = kwargs.get("name", None)
        extras = kwargs.get("extras", {})
        return cls(*others, name=name, extras=dict(extras,
            _uniform=Uniform.of(*others, name="%sUniform" % name if name else name, extras=extras)
        ))


# example extension classes
class Lorentz(object):
    def boost(self, vec):
        return -1


class LorentzXYZE(Lorentz):
    @cachedProperty
    def pt(self):
        return self.px**2 + self.py**2


class LorentzCyl(Lorentz):
    @cachedProperty
    def px(self):
        return self.pt * self._backend.sin(self.phi)


class Muon(object):
    @cachedProperty
    def good_id(self):
        return self.uncorr.pt > 50 & self.iso < 0.2 # <----- CRUX


class Electron(object):
    @cachedProperty
    def caloCluster(self):
        return self._sub(
            LorentzCyl,
            _links=dict(
                eta="Electron_caloClusterEta",
                phi="Electron_caloClusterPhi",
                pt="Electron_caloClusterPt",
            )
        )


class Event(object):
    _runIdMin = np.array([1, 200, 5000])
    _runIdMax = np.array([5, 300, 5123])

    @cachedProperty
    def good(self):
        return (
            (self._runIdMin <= self.runId[..., None]) &
            (self.runId[..., None] < self._runIdMax)
        ).any(axis=-1)

# example use case 1
base = 123

ev = Uniform.of(Event)(
    _base=base,
    _prefix="data",
    _missing=("npyfile", "event"),
    _links=dict(
        id="eventId",
        runId="runId",
    ),
    mu=Jagged.of(Muon, LorentzCyl)(
        _base=base,
        _missing=("npyfile", "muon"),
        _sizes=Trace("data", "nMuon"),
        _prefix=("data"),
        _links=dict(
            iso="Muon_IsolationFooBARRRR",
            pt="Muon_Pt",
            eta="Muon_Eta",
            phi="Muon_Phi",
        )
    ),
    el=Jagged.of(Electron, LorentzXYZE)(
        _base=base,
        _sizes=Trace("data", "nElectron"),
        _prefix=Trace("data"),
        _links=dict(
            px="Electron_Px",
            py="Electron_Py",
            pz="Electron_Pz",
            e="Electron_E",
        )
    )
)

ev.mu.uncorr = ev.mu._sub(
    pt="Muon_PtUncorr",
    eta="Muon_Eta",
    phi="Muon_Phi",
)
ev.mu.uncorr = ev.mu._ext(
    pt="Muon_PtUncorr",
)

# example use case 2
q = Jagged(
    px=returnNpyishArray("px"),
    py=returnNpyishArray("py"),
    pz=returnNpyishArray("pz"),
)
q.pq = np.empty_like(q.px)

q.foo = q.px < q.py
q.foo[p.pz > 0] = q.px > q.pz

with q[q.foo] as qq:
    qq.pq = qq.px * qq.py
with q[~q.foo] as qq:
    qq.pq = qq.px * qq.px - qq.pz

# example use case 3
ev = Uproot( # auto: group, lorentz, cluster
    data,
    Muon=Muon,
    Electron=Electron,
)

ev.el[:, :2].px**2 > 15

### very old doodles
ev.jets = jets = JaggedGroup(tree_chunk)

jets.good = jets._LOCAL

jets.btag_sf = SF_calc(jets)
jets.good = jets.pt > 30 & jets.puId == 7 & jets.jetId == 3
jets.tagged = jets.good & jets.btagCMVA > 0.4

ev.nGood = jets[..., :].good.sum(axis=-1)
ev.nGood = jets._sum(jets.good, axis=-1, keepdims=True) # same for prod

# reduce
jets[jets.good]
jets[ev.nGood > 4, :]
jets[:, :4]

jets[ev.nGood > 4]
