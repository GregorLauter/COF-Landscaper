#!/usr/bin/env python3
"""
Batch CIF → CRYSTAL .d12 converter
----------------------------------
Converts all .cif files in the current directory into .d12 files.
Takes unit cell and fractional coordinates directly from each CIF.

Set MODE at the top of the file to choose which POST_BLOCK to use.
"""

import re
from pathlib import Path
from typing import Optional

# ===== USER-EDITABLE BLOCKS =====
# 1 → use OPT (POST_BLOCK_1)
# 2 → use SP (POST_BLOCK_2)
MODE = 1

POST_BLOCK_1 = """OPTGEOM
MAXTRADIUS
1.0
ENDOPT
BASISSET
SOLDEF2MSVP
DFT
HSESOL3C
END
SHRINK
0 8
2 2 8
END"""

POST_BLOCK_2 = """BASISSET
SOLDEF2MSVP
DFT
HSESOL3C
END
SHRINK
0 8
2 2 8
END"""

# Automatically choose the active block based on MODE
if MODE == 1:
    POST_BLOCK = POST_BLOCK_1
elif MODE == 2:
    POST_BLOCK = POST_BLOCK_2
else:
    raise ValueError(f"Unsupported MODE={MODE}. Use 1 or 2.")

# =================================


PERIODIC = {
'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,
'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,'Pa':91,'U':92
}

def strip_esd(val: str) -> str:
    return re.sub(r"\([^)]*\)", "", val)

def parse_float(val: str) -> float:
    return float(strip_esd(val))

def guess_symbol(raw: str) -> Optional[str]:
    s = re.sub(r"[^A-Za-z]", "", raw)
    if not s: return None
    s = s[0].upper() + s[1:].lower()
    if s in PERIODIC: return s
    if len(s)>1 and s[:2] in PERIODIC: return s[:2]
    if s[0] in PERIODIC: return s[0]
    return None

def parse_cell(text: str):
    def grab(key):
        m = re.search(rf"{key}\s+([0-9.+\-Ee()]+)", text)
        if not m: raise ValueError(f"Missing cell parameter: {key}")
        return parse_float(m.group(1))
    return dict(
        a=grab(r"_cell_length_a"),
        b=grab(r"_cell_length_b"),
        c=grab(r"_cell_length_c"),
        alpha=grab(r"_cell_angle_alpha"),
        beta=grab(r"_cell_angle_beta"),
        gamma=grab(r"_cell_angle_gamma")
    )

def find_atom_loop(lines):
    for i,line in enumerate(lines):
        if line.strip().lower().startswith("loop_"):
            j=i+1; headers=[]
            while j<len(lines) and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip()); j+=1
            if any("fract_x" in h.lower() for h in headers):
                return headers,j
    raise ValueError("No atom site loop with fractional coordinates.")

def extract_atoms(lines, headers, start):
    hdr={h.split()[0].lower():i for i,h in enumerate(headers)}
    xk=next(k for k in hdr if "_fract_x" in k)
    yk=next(k for k in hdr if "_fract_y" in k)
    zk=next(k for k in hdr if "_fract_z" in k)
    lbl=next((k for k in hdr if "_type_symbol" in k or "_label" in k), None)
    atoms=[]; j=start
    while j<len(lines):
        s=lines[j].strip()
        if not s or s.startswith("loop_") or s.startswith("_"): break
        p=lines[j].split()
        if len(p)>=len(headers):
            lab=p[hdr[lbl]] if lbl else "X"
            sym=guess_symbol(lab) or "X"
            Z=PERIODIC.get(sym,-1)
            x=parse_float(p[hdr[xk]]); y=parse_float(p[hdr[yk]]); z=parse_float(p[hdr[zk]])
            atoms.append((Z,x,y,z))
        j+=1
    return atoms

def convert_one(cif_path: Path):
    txt=cif_path.read_text(errors="ignore")
    lines=txt.splitlines()
    cell=parse_cell(txt)
    headers,start=find_atom_loop(lines)
    atoms=extract_atoms(lines,headers,start)
    title=cif_path.stem
    out_path=cif_path.with_suffix(".d12")

    out=[]
    out.append(title)
    out.append("CRYSTAL")
    out.append("0 0 0")
    out.append("1")  # P1 symmetry
    out.append(f"{cell['a']:.6f} {cell['b']:.6f} {cell['c']:.6f} {cell['alpha']:.6f} {cell['beta']:.6f} {cell['gamma']:.6f}")
    out.append(str(len(atoms)))
    for Z,x,y,z in atoms:
        if Z<0: out.append(f"0 {x:.9f} {y:.9f} {z:.9f}")
        else:   out.append(f"{Z} {x:.9f} {y:.9f} {z:.9f}")
    if POST_BLOCK.strip(): out.append(POST_BLOCK.strip())
    out_path.write_text("\n".join(out)+"\n")
    print(f"✔ Wrote {out_path.name}")

def main():
    for cif in Path(".").glob("*.cif"):
        try:
            convert_one(cif)
        except Exception as e:
            print(f"✖ Failed {cif.name}: {e}")

if __name__=="__main__":
    main()