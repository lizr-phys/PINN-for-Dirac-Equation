from pathlib import Path
import shutil
from plan import BASE,PROJECT,write,sha
target=BASE/'closure/vendor';target.mkdir(parents=True,exist_ok=True)
records=[]
for p in sorted((PROJECT/'src').glob('*.py')):
    q=target/p.name
    if q.exists():
        if sha(p)!=sha(q):raise RuntimeError('Vendor snapshot mismatch '+str(p))
    else:shutil.copyfile(p,q)
    records.append({'source':str(p.relative_to(PROJECT)),'copy':str(q.relative_to(BASE)),'sha256':sha(p)})
(BASE/'closure/__init__.py').touch(exist_ok=True)
(target/'__init__.py').touch(exist_ok=True)
write(BASE/'history/vendor_source_snapshot.json',records)
print('Read-only historical source snapshot',len(records))
