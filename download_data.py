import sys, json, re, time, hashlib, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parent/'download-libs'))
import gdown

ROOT=Path(__file__).resolve().parent/'data'; ROOT.mkdir(exist_ok=True,parents=True)
def clinic():
    manifest=ROOT/'clinic_drive_manifest.json'
    if manifest.exists():
        files=json.loads(manifest.read_text())
    else:
        raw=gdown.download_folder(id='1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P',output=str(ROOT/'ClinicDB')+'/',skip_download=True,quiet=True)
        files=[dict(id=x.id,path=x.path,local_path=x.local_path) for x in raw]
        manifest.write_text(json.dumps(files,indent=2))
    def download(x):
        p=Path(x['local_path']); p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.stat().st_size>0:return
        for trial in range(4):
            try:
                url='https://drive.google.com/uc?export=download&id='+x['id']
                with urllib.request.urlopen(url,timeout=20) as response:
                    data=response.read()
                if p.suffix=='.png' and not data.startswith(b'\x89PNG'):raise RuntimeError('Non-image response')
                p.write_bytes(data)
                return
            except Exception as e:
                if trial==3: raise
                time.sleep(2*(trial+1))
        raise RuntimeError(str(p))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(download,x) for x in files]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%100==0:print(f'ClinicDB {i}/{len(files)}',flush=True)
    print('ClinicDB complete',len(files),flush=True)

def mass():
    base='https://www.cs.toronto.edu/~vmnih/data/mass_buildings/'
    files=[]
    for split in ['train','valid','test']:
        for kind in ['sat','map']:
            url=base+split+'/'+kind+'/index.html'
            raw=urllib.request.urlopen(url,timeout=60).read().decode()
            names=sorted(set(re.findall(r'>([^<>]+\.tiff?)</a>',raw)))
            print('Massachusetts index',split,kind,len(names),flush=True)
            for name in names:files.append(dict(url=base+split+'/'+kind+'/'+name,path=str(ROOT/'Massachusetts'/split/kind/name)))
    (ROOT/'mass_download_manifest.json').write_text(json.dumps(files,indent=2))
    def download(x):
        p=Path(x['path']);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.stat().st_size>0:return
        for trial in range(4):
            try:
                with urllib.request.urlopen(x['url'],timeout=20) as r:
                    p.write_bytes(r.read())
                return
            except Exception as e:
                print('Retry',p.name,str(e),flush=True)
                if trial==3:raise
                time.sleep(2*(trial+1))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download,x) for x in files]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%25==0:print(f'Massachusetts {i}/{len(files)}',flush=True)
    print('Massachusetts complete',len(files),flush=True)

if __name__=='__main__':
    {'clinic':clinic,'mass':mass}[sys.argv[1]]()
