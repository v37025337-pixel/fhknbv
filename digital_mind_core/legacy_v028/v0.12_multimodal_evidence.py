
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import hashlib, ipaddress, re

class Domain(str, Enum):
    NETWORK_CONFIG="network_config"
    SECURITY_EVENT="security_event"
    DOCUMENT="document"
    UNKNOWN="unknown"

@dataclass
class Observation:
    key:str
    value:Any
    confidence:float=1.0
    sensitive:bool=False

@dataclass
class Entity:
    eid:str
    kind:str
    attrs:Dict[str,Any]=field(default_factory=dict)

@dataclass
class Relation:
    src:str
    rel:str
    dst:str
    confidence:float=1.0
    evidence_id:str|None=None

@dataclass
class Constraint:
    kind:str
    expression:str
    evidence_id:str|None=None

@dataclass
class EvidenceRef:
    evidence_id:str
    path:str
    sha256:str
    media_type:str="image"

class EvidenceWorld:
    def __init__(self):
        self.entities={}
        self.relations=[]
        self.constraints=[]
        self.evidence={}

    def add_entity(self,e): self.entities[e.eid]=e
    def add_relation(self,r): self.relations.append(r)
    def add_constraint(self,c): self.constraints.append(c)

    def stats(self):
        kinds={}
        rels={}
        for e in self.entities.values():
            kinds[e.kind]=kinds.get(e.kind,0)+1
        for r in self.relations:
            rels[r.rel]=rels.get(r.rel,0)+1
        return {
            "evidence_items":len(self.evidence),
            "entities":len(self.entities),
            "relations":len(self.relations),
            "constraints":len(self.constraints),
            "entity_kinds":kinds,
            "relation_kinds":rels,
        }

class EvidenceStore:
    def __init__(self,world): self.world=world

    def register(self,path):
        raw=Path(path).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        ref=EvidenceRef("evidence:"+digest[:20], str(path), digest)
        self.world.evidence[ref.evidence_id]=ref
        return ref

class EvidenceRouter:
    def route(self,obs:List[Observation]):
        keys={o.key for o in obs}
        dns=len(keys & {"provider","primary_dns","secondary_dns"})
        sec=len(keys & {"service","event_type","location","time","browser","os","ip","account"})
        if dns>=2 and ("primary_dns" in keys or "secondary_dns" in keys):
            return Domain.NETWORK_CONFIG, min(.99,.60+.10*dns)
        if sec>=3 and ("service" in keys or "event_type" in keys):
            return Domain.SECURITY_EVENT, min(.99,.50+.06*sec)
        if keys:
            return Domain.DOCUMENT,.55
        return Domain.UNKNOWN,0.0

def valid_ip(x):
    try:
        ipaddress.ip_address(x); return True
    except Exception:
        return False

class MultiDomainEvidenceKernel:
    def __init__(self):
        self.world=EvidenceWorld()
        self.store=EvidenceStore(self.world)
        self.router=EvidenceRouter()

    def ingest_dns_image(self,path,rows):
        ref=self.store.register(path)
        obs=[Observation("provider","table"),Observation("primary_dns","column"),Observation("secondary_dns","column")]
        domain,conf=self.router.route(obs)
        if domain!=Domain.NETWORK_CONFIG: raise RuntimeError("bad route")
        table=f"dns_table:{ref.evidence_id}"
        self.world.add_entity(Entity(table,"dns_table",{"rows":len(rows)}))
        for row in rows:
            provider=row["provider"].strip()
            p=row["primary_dns"].strip()
            s=row["secondary_dns"].strip()
            if not(valid_ip(p) and valid_ip(s)):
                self.world.add_constraint(Constraint("validation",f"invalid DNS row: {provider}",ref.evidence_id))
                continue
            pe="dns_provider:"+provider.lower().replace(" ","_")
            p1="dns_server:"+p
            p2="dns_server:"+s
            self.world.add_entity(Entity(pe,"dns_provider",{"name":provider}))
            self.world.add_entity(Entity(p1,"dns_server",{"address":p}))
            self.world.add_entity(Entity(p2,"dns_server",{"address":s}))
            self.world.add_relation(Relation(table,"LISTS_PROVIDER",pe,1,ref.evidence_id))
            self.world.add_relation(Relation(pe,"PRIMARY_DNS",p1,1,ref.evidence_id))
            self.world.add_relation(Relation(pe,"SECONDARY_DNS",p2,1,ref.evidence_id))
        return domain,conf

    def ingest_security_image(self,path,obs):
        ref=self.store.register(path)
        domain,conf=self.router.route(obs)
        if domain!=Domain.SECURITY_EVENT: raise RuntimeError("bad route")
        by={o.key:o for o in obs}
        attrs={}
        for k,o in by.items():
            attrs[k]={"value":o.value,"sensitive":True,"display":"<redacted>"} if o.sensitive else o.value
        event=f"security_event:{ref.evidence_id}"
        self.world.add_entity(Entity(event,"security_event",attrs))
        service=str(by.get("service",Observation("service","unknown")).value)
        service_id="service:"+service.lower()
        self.world.add_entity(Entity(service_id,"service",{"name":service}))
        self.world.add_relation(Relation(event,"OCCURRED_ON_SERVICE",service_id,1,ref.evidence_id))
        if "browser" in by:
            b="browser:"+str(by["browser"].value).lower().replace(" ","_")
            self.world.add_entity(Entity(b,"browser",{"name":by["browser"].value}))
            self.world.add_relation(Relation(event,"USED_BROWSER",b,by["browser"].confidence,ref.evidence_id))
        if "os" in by:
            o="os:"+str(by["os"].value).lower().replace(" ","_")
            self.world.add_entity(Entity(o,"os",{"name":by["os"].value}))
            self.world.add_relation(Relation(event,"USED_OS",o,by["os"].confidence,ref.evidence_id))
        for k in ("ip","account","location"):
            if k in by and not by[k].sensitive:
                self.world.add_constraint(Constraint("privacy",f"{k} should be sensitive",ref.evidence_id))
        et=str(by.get("event_type",Observation("event_type","event")).value)
        if re.search(r"sign.?in|login",et,re.I):
            self.world.add_constraint(Constraint("epistemic","sign-in alert requires verification; do not infer compromise from alert alone",ref.evidence_id))
        return domain,conf

    def safe_summary(self):
        sensitive=0
        for e in self.world.entities.values():
            if e.kind=="security_event":
                sensitive += sum(isinstance(v,dict) and v.get("sensitive") for v in e.attrs.values())
        s=self.world.stats()
        s["sensitive_fields"]=sensitive
        s["raw_evidence_preserved"]=len(self.world.evidence)
        return s
