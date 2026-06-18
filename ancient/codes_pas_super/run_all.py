"""run_all.py — lance toutes les méthodes."""
from __future__ import annotations
import argparse, importlib, sys, time
from pathlib import Path

MODULES = {
    "01":{"file":"01_background_subtraction","label":"Soustraction_de_fond","methods":["mog2","knn","vibe"]},
    "02":{"file":"02_segmentation","label":"Segmentation","methods":["log","chanvese"]},
    "03":{"file":"03_blob_detection","label":"Detection_blobs","methods":["log","dog","doh","simpleblob"]},
    "04":{"file":"04_tracking_association","label":"Tracking","methods":["sort","trackpy","farneback"]},
    "05":{"file":"05_merge_split","label":"Fusions_Separations","methods":["topological","laptrack"]},
}
METHOD_FN={
    "mog2":"run_mog2","knn":"run_knn","vibe":"run_vibe",
    "log":"run_log_nms","chanvese":"run_chanvese_blur",
    "dog":"run_dog","doh":"run_doh","simpleblob":"run_simple_blob",
    "sort":"run_sort","trackpy":"run_trackpy","farneback":"run_farneback_guided",
    "topological":"run_topological","laptrack":"run_laptrack",
}

def run_all(video_path,fps_extract,modules_to_run,method_filter,output_root):
    here=Path(__file__).resolve().parent
    if str(here) not in sys.path: sys.path.insert(0,str(here))
    from utils import load_frames,save_video
    print(f"\n{'='*55}\n  Vidéo : {video_path.name}  fps_extract={fps_extract}\n{'='*55}\n")
    frames,src_fps,src_dur=load_frames(video_path,fps_extract=fps_extract)
    print(f"  {len(frames)} frames | {frames[0].shape} | {src_fps:.1f}fps | {src_dur:.1f}s\n")
    summary=[]
    for mod_id in modules_to_run:
        if mod_id not in MODULES: continue
        info=MODULES[mod_id]; mod=importlib.import_module(info["file"])
        out_dir=output_root/f"{mod_id}_{info['label']}"; out_dir.mkdir(parents=True,exist_ok=True)
        methods=[method_filter] if method_filter and method_filter in info["methods"] else info["methods"]
        print(f"\n{'─'*48}\n  MODULE {mod_id} — {info['label']}\n{'─'*48}")
        for method in methods:
            fn=getattr(mod,METHOD_FN.get(method,""),None)
            if fn is None: print(f"  [SKIP] {method}"); continue
            out_path=out_dir/f"{mod_id}_{method}.mp4"; print(f"\n  ► {method.upper()}")
            t0=time.perf_counter()
            try:
                ann=fn(frames); save_video(ann,out_path,src_duration=src_dur)
                e=time.perf_counter()-t0; summary.append((mod_id,method,len(ann),e,out_path.name)); print(f"    ✓ {e:.1f}s")
            except Exception as ex:
                e=time.perf_counter()-t0; print(f"    ✗ {ex}"); summary.append((mod_id,method,0,e,str(ex)))
    print(f"\n{'='*55}  RÉCAP")
    for row in summary:
        print(f"  {row[0]} {row[1]:<16} {row[2]:5d}fr  {row[3]:5.1f}s  {row[4]}")
    print(f"{'='*55}\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video",type=Path)
    ap.add_argument("--fps-extract",type=float,default=1.0)
    ap.add_argument("--modules",nargs="*",default=list(MODULES))
    ap.add_argument("--method",type=str,default=None)
    ap.add_argument("--output-dir",type=Path,default=None)
    args=ap.parse_args()
    out=args.output_dir or args.video.parent/f"{args.video.stem}_results"
    run_all(args.video,args.fps_extract,args.modules,args.method,out)

if __name__=="__main__": main()