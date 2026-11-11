"""Comprehensive Pharmacopeia Database for Indian & International Medical Prescriptions."""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional


@dataclass
class MedicineInfo:
    standard_name: str
    category: str
    common_formulations: List[str]
    common_dosages: List[str]
    brand_aliases: List[str]
    soundalike_warnings: List[str]


# Indian & International standard formulary mapping
PHARMACOPEIA: Dict[str, MedicineInfo] = {
    # Analgesics & Antipyretics
    "paracetamol": MedicineInfo(
        standard_name="Paracetamol",
        category="Analgesic / Antipyretic",
        common_formulations=["TAB", "SYP", "INJ", "SUPP"],
        common_dosages=["500 MG", "650 MG", "1000 MG", "120 MG/5ML", "250 MG/5ML"],
        brand_aliases=["pcm", "dolo", "calpol", "crocin", "pacimol", "pyregesic", "fepex", "acetaminophen", "aectaminepen", "tylenol"],
        soundalike_warnings=["paraquat"],
    ),
    "ibuprofen": MedicineInfo(
        standard_name="Ibuprofen",
        category="NSAID",
        common_formulations=["TAB", "SYP"],
        common_dosages=["200 MG", "400 MG", "600 MG", "800 MG"],
        brand_aliases=["brufen", "combiflam", "ibugesic"],
        soundalike_warnings=[],
    ),
    "diclofenac": MedicineInfo(
        standard_name="Diclofenac",
        category="NSAID",
        common_formulations=["TAB", "INJ", "GEL"],
        common_dosages=["50 MG", "75 MG", "100 MG"],
        brand_aliases=["voveran", "dynapar", "voltaren", "diclogesic"],
        soundalike_warnings=["diflucan"],
    ),
    "tramadol": MedicineInfo(
        standard_name="Tramadol",
        category="Opioid Analgesic",
        common_formulations=["TAB", "CAP", "INJ"],
        common_dosages=["50 MG", "100 MG"],
        brand_aliases=["ultram", "tramazac", "supridol"],
        soundalike_warnings=["trazodone", "toradol"],
    ),
    "aceclofenac": MedicineInfo(
        standard_name="Aceclofenac",
        category="NSAID",
        common_formulations=["TAB"],
        common_dosages=["100 MG", "200 MG"],
        brand_aliases=["hifenac", "zerodol", "acemiz"],
        soundalike_warnings=[],
    ),

    # Antibiotics & Antimicrobials
    "amoxicillin": MedicineInfo(
        standard_name="Amoxicillin",
        category="Antibiotic (Penicillin)",
        common_formulations=["CAP", "TAB", "SYP"],
        common_dosages=["250 MG", "500 MG", "625 MG", "875 MG", "125 MG/5ML"],
        brand_aliases=["amoxil", "augmentin", "moxikind", "mox", "novamox", "clamox", "aoicillin", "amoxcilln", "amox"],
        soundalike_warnings=["ampicillin", "amoxapine"],
    ),
    "azithromycin": MedicineInfo(
        standard_name="Azithromycin",
        category="Antibiotic (Macrolide)",
        common_formulations=["TAB", "SYP", "INJ"],
        common_dosages=["250 MG", "500 MG", "100 MG/5ML", "200 MG/5ML"],
        brand_aliases=["azithral", "zithromax", "aziwok", "azee"],
        soundalike_warnings=["erythromycin", "clarithromycin"],
    ),
    "ciprofloxacin": MedicineInfo(
        standard_name="Ciprofloxacin",
        category="Antibiotic (Fluoroquinolone)",
        common_formulations=["TAB", "EYE DROPS", "INJ"],
        common_dosages=["250 MG", "500 MG", "750 MG"],
        brand_aliases=["cipro", "ciptox", "cifran", "ciplox"],
        soundalike_warnings=["cephalexin"],
    ),
    "cefixime": MedicineInfo(
        standard_name="Cefixime",
        category="Antibiotic (Cephalosporin)",
        common_formulations=["TAB", "SYP"],
        common_dosages=["100 MG", "200 MG", "400 MG"],
        brand_aliases=["zifi", "taxim-o", "cefolac", "suprax"],
        soundalike_warnings=["cefuroxime", "cefaclor"],
    ),
    "doxycycline": MedicineInfo(
        standard_name="Doxycycline",
        category="Antibiotic (Tetracycline)",
        common_formulations=["CAP", "TAB"],
        common_dosages=["100 MG", "200 MG"],
        brand_aliases=["doxypal", "vibramycin", "monodox", "dox-1"],
        soundalike_warnings=["dicyclomine"],
    ),
    "metronidazole": MedicineInfo(
        standard_name="Metronidazole",
        category="Antiprotozoal / Antibacterial",
        common_formulations=["TAB", "INJ", "GEL"],
        common_dosages=["200 MG", "400 MG", "500 MG"],
        brand_aliases=["flagyl", "metrogyl"],
        soundalike_warnings=["metformin"],
    ),

    # Gastrointestinal / Acid Reducers (PPIs & H2 blockers)
    "pantoprazole": MedicineInfo(
        standard_name="Pantoprazole",
        category="Proton Pump Inhibitor",
        common_formulations=["TAB", "INJ"],
        common_dosages=["20 MG", "40 MG"],
        brand_aliases=["pan", "pantocid", "pantosec", "protonix"],
        soundalike_warnings=[],
    ),
    "omeprazole": MedicineInfo(
        standard_name="Omeprazole",
        category="Proton Pump Inhibitor",
        common_formulations=["CAP", "TAB"],
        common_dosages=["10 MG", "20 MG", "40 MG"],
        brand_aliases=["prilosec", "omez", "ocid"],
        soundalike_warnings=[],
    ),
    "rabeprazole": MedicineInfo(
        standard_name="Rabeprazole",
        category="Proton Pump Inhibitor",
        common_formulations=["TAB"],
        common_dosages=["10 MG", "20 MG"],
        brand_aliases=["rabicip", "rabeloc", "aciphex", "rabiros"],
        soundalike_warnings=[],
    ),
    "domperidone": MedicineInfo(
        standard_name="Domperidone",
        category="Antiemetic / Prokinetic",
        common_formulations=["TAB", "SYP"],
        common_dosages=["10 MG", "30 MG"],
        brand_aliases=["motilium", "vomistop", "domstal"],
        soundalike_warnings=[],
    ),
    "cimetidine": MedicineInfo(
        standard_name="Cimetidine",
        category="H2 Receptor Antagonist",
        common_formulations=["TAB", "INJ"],
        common_dosages=["50 MG", "100 MG", "200 MG", "400 MG", "800 MG"],
        brand_aliases=["tagamet", "cimet", "cimetiline"],
        soundalike_warnings=[],
    ),
    "ondansetron": MedicineInfo(
        standard_name="Ondansetron",
        category="Antiemetic",
        common_formulations=["TAB", "SYP", "INJ"],
        common_dosages=["4 MG", "8 MG"],
        brand_aliases=["zofran", "emset", "vomikind"],
        soundalike_warnings=[],
    ),

    # Cardiovascular & Antihypertensives
    "amlodipine": MedicineInfo(
        standard_name="Amlodipine",
        category="Calcium Channel Blocker",
        common_formulations=["TAB"],
        common_dosages=["2.5 MG", "5 MG", "10 MG"],
        brand_aliases=["norvasc", "amlong", "stamlo", "amlovas"],
        soundalike_warnings=["amiloride"],
    ),
    "telmisartan": MedicineInfo(
        standard_name="Telmisartan",
        category="Angiotensin Receptor Blocker",
        common_formulations=["TAB"],
        common_dosages=["20 MG", "40 MG", "80 MG"],
        brand_aliases=["micardis", "telma", "telpres", "telsartan"],
        soundalike_warnings=[],
    ),
    "losartan": MedicineInfo(
        standard_name="Losartan",
        category="Angiotensin Receptor Blocker",
        common_formulations=["TAB"],
        common_dosages=["25 MG", "50 MG", "100 MG"],
        brand_aliases=["cozaar", "losacar", "repace", "tozaar", "jesahan", "lesaman"],
        soundalike_warnings=["valsartan"],
    ),
    "lisinopril": MedicineInfo(
        standard_name="Lisinopril",
        category="ACE Inhibitor",
        common_formulations=["TAB"],
        common_dosages=["2.5 MG", "5 MG", "10 MG", "20 MG", "40 MG", "250 MG"],
        brand_aliases=["prinivil", "zestril", "listril", "lisinapl"],
        soundalike_warnings=["risperidone"],
    ),
    "hydrochlorothiazide": MedicineInfo(
        standard_name="Hydrochlorothiazide",
        category="Diuretic",
        common_formulations=["TAB"],
        common_dosages=["12.5 MG", "25 MG", "50 MG", "5 MG", "250 MG"],
        brand_aliases=["hctz", "aquazide", "microzide", "hyehoechertlazik", "hyehocerotiazit"],
        soundalike_warnings=[],
    ),
    "atenolol": MedicineInfo(
        standard_name="Atenolol",
        category="Beta Blocker",
        common_formulations=["TAB"],
        common_dosages=["25 MG", "50 MG", "100 MG"],
        brand_aliases=["tenormin", "aten"],
        soundalike_warnings=["albuterol"],
    ),
    "metoprolol": MedicineInfo(
        standard_name="Metoprolol",
        category="Beta Blocker",
        common_formulations=["TAB"],
        common_dosages=["25 MG", "50 MG", "100 MG"],
        brand_aliases=["lopressor", "toprol-xl", "metolar", "betaloc"],
        soundalike_warnings=[],
    ),
    "oxprenolol": MedicineInfo(
        standard_name="Oxprenolol",
        category="Beta Blocker",
        common_formulations=["TAB"],
        common_dosages=["20 MG", "40 MG", "50 MG", "80 MG", "160 MG"],
        brand_aliases=["trasicor", "oxprelol"],
        soundalike_warnings=[],
    ),
    "dorzolamide": MedicineInfo(
        standard_name="Dorzolamide",
        category="Carbonic Anhydrase Inhibitor",
        common_formulations=["DROPS", "TAB"],
        common_dosages=["10 MG", "20 MG", "2%"],
        brand_aliases=["trusopt", "dorzox", "dorzolamidum"],
        soundalike_warnings=[],
    ),

    # Antidiabetics
    "metformin": MedicineInfo(
        standard_name="Metformin",
        category="Biguanide",
        common_formulations=["TAB"],
        common_dosages=["500 MG", "850 MG", "1000 MG", "5 MG"],
        brand_aliases=["glucophage", "glycomet", "gluformin", "obimet", "meitermin", "yneormn"],
        soundalike_warnings=["metronidazole"],
    ),
    "glimepiride": MedicineInfo(
        standard_name="Glimepiride",
        category="Sulfonylurea",
        common_formulations=["TAB"],
        common_dosages=["1 MG", "2 MG", "3 MG", "4 MG"],
        brand_aliases=["amaryl", "glypride", "amaryl-m"],
        soundalike_warnings=["glipizide"],
    ),
    "vildagliptin": MedicineInfo(
        standard_name="Vildagliptin",
        category="DPP-4 Inhibitor",
        common_formulations=["TAB"],
        common_dosages=["50 MG", "100 MG"],
        brand_aliases=["galvus", "jalra", "vildaprime"],
        soundalike_warnings=[],
    ),
    "dapagliflozin": MedicineInfo(
        standard_name="Dapagliflozin",
        category="SGLT2 Inhibitor",
        common_formulations=["TAB"],
        common_dosages=["5 MG", "10 MG"],
        brand_aliases=["forxiga", "dapa", "oxra"],
        soundalike_warnings=[],
    ),

    # Statins / Lipid Lowering
    "atorvastatin": MedicineInfo(
        standard_name="Atorvastatin",
        category="HMG-CoA Reductase Inhibitor (Statin)",
        common_formulations=["TAB"],
        common_dosages=["5 MG", "10 MG", "20 MG", "40 MG", "80 MG"],
        brand_aliases=["lipitor", "atorva", "atocor", "storvas", "aorastatin"],
        soundalike_warnings=[],
    ),
    "rosuvastatin": MedicineInfo(
        standard_name="Rosuvastatin",
        category="Statin",
        common_formulations=["TAB"],
        common_dosages=["5 MG", "10 MG", "20 MG", "40 MG"],
        brand_aliases=["crestor", "rosuvas", "novastat", "rosulip"],
        soundalike_warnings=[],
    ),
    "simvastatin": MedicineInfo(
        standard_name="Simvastatin",
        category="Statin",
        common_formulations=["TAB"],
        common_dosages=["10 MG", "20 MG", "25 MG", "40 MG"],
        brand_aliases=["zocor", "simvotin", "simcard", "simoagtain"],
        soundalike_warnings=[],
    ),

    # Antiallergic & Respiratory
    "cetirizine": MedicineInfo(
        standard_name="Cetirizine",
        category="Antihistamine",
        common_formulations=["TAB", "SYP"],
        common_dosages=["5 MG", "10 MG", "5 MG/5ML"],
        brand_aliases=["zyrtec", "cetzine", "okacet", "alercet"],
        soundalike_warnings=[],
    ),
    "levocetirizine": MedicineInfo(
        standard_name="Levocetirizine",
        category="Antihistamine",
        common_formulations=["TAB", "SYP"],
        brand_aliases=["xyzal", "levocet", "teczine"],
        common_dosages=["2.5 MG", "5 MG"],
        soundalike_warnings=[],
    ),
    "montelukast": MedicineInfo(
        standard_name="Montelukast",
        category="Leukotriene Receptor Antagonist",
        common_formulations=["TAB"],
        common_dosages=["4 MG", "5 MG", "10 MG"],
        brand_aliases=["singulair", "montek", "montair"],
        soundalike_warnings=[],
    ),
    "salbutamol": MedicineInfo(
        standard_name="Salbutamol",
        category="Beta-2 Agonist (Bronchodilator)",
        common_formulations=["INH", "SYP", "TAB"],
        common_dosages=["2 MG", "4 MG", "100 MCG"],
        brand_aliases=["ventolin", "asthalin"],
        soundalike_warnings=[],
    ),

    # Corticosteroids
    "prednisone": MedicineInfo(
        standard_name="Prednisone",
        category="Corticosteroid",
        common_formulations=["TAB"],
        common_dosages=["5 MG", "10 MG", "20 MG", "25 MG", "50 MG"],
        brand_aliases=["deltasone", "prednicot", "prchusene"],
        soundalike_warnings=["prednisolone"],
    ),
    "prednisolone": MedicineInfo(
        standard_name="Prednisolone",
        category="Corticosteroid",
        common_formulations=["TAB", "SYP", "EYE DROPS"],
        common_dosages=["5 MG", "10 MG", "20 MG"],
        brand_aliases=["omnipred", "wysolone"],
        soundalike_warnings=["prednisone"],
    ),
    "dexamethasone": MedicineInfo(
        standard_name="Dexamethasone",
        category="Corticosteroid",
        common_formulations=["TAB", "INJ"],
        common_dosages=["0.5 MG", "2 MG", "4 MG", "8 MG"],
        brand_aliases=["decadron", "dexona"],
        soundalike_warnings=[],
    ),

    # Additional Essential Antibiotics & Antifungals
    "levofloxacin": MedicineInfo(
        standard_name="Levofloxacin",
        category="Antibiotic (Fluoroquinolone)",
        common_formulations=["TAB", "INJ", "EYE DROPS"],
        common_dosages=["250 MG", "500 MG", "750 MG"],
        brand_aliases=["levomac", "loxof", "glevo"],
        soundalike_warnings=[],
    ),
    "ofloxacin": MedicineInfo(
        standard_name="Ofloxacin",
        category="Antibiotic (Fluoroquinolone)",
        common_formulations=["TAB", "EYE DROPS"],
        common_dosages=["200 MG", "400 MG"],
        brand_aliases=["oflox", "zanocin", "oflamac"],
        soundalike_warnings=[],
    ),
    "ceftriaxone": MedicineInfo(
        standard_name="Ceftriaxone",
        category="Antibiotic (Cephalosporin)",
        common_formulations=["INJ"],
        common_dosages=["250 MG", "500 MG", "1000 MG", "1 GM", "2 GM"],
        brand_aliases=["monocef", "rocephin"],
        soundalike_warnings=[],
    ),
    "fluconazole": MedicineInfo(
        standard_name="Fluconazole",
        category="Antifungal",
        common_formulations=["TAB", "CAP"],
        common_dosages=["50 MG", "150 MG", "200 MG", "400 MG"],
        brand_aliases=["diflucan", "forcan", "syscan"],
        soundalike_warnings=["diclofenac"],
    ),

    # Additional Gastrointestinal & Antiemetics
    "esomeprazole": MedicineInfo(
        standard_name="Esomeprazole",
        category="Proton Pump Inhibitor",
        common_formulations=["TAB", "INJ"],
        common_dosages=["20 MG", "40 MG"],
        brand_aliases=["nexpro", "nexium", "esomac"],
        soundalike_warnings=[],
    ),
    "ranitidine": MedicineInfo(
        standard_name="Ranitidine",
        category="H2 Receptor Antagonist",
        common_formulations=["TAB", "INJ", "SYP"],
        common_dosages=["150 MG", "300 MG"],
        brand_aliases=["zinetac", "rantac", "zantac"],
        soundalike_warnings=[],
    ),

    # Additional Cardiovascular
    "ramipril": MedicineInfo(
        standard_name="Ramipril",
        category="ACE Inhibitor",
        common_formulations=["TAB", "CAP"],
        common_dosages=["2.5 MG", "5 MG", "10 MG"],
        brand_aliases=["cardace", "altace", "ramace"],
        soundalike_warnings=[],
    ),
    "propranolol": MedicineInfo(
        standard_name="Propranolol",
        category="Beta Blocker",
        common_formulations=["TAB", "INJ"],
        common_dosages=["10 MG", "20 MG", "40 MG", "80 MG"],
        brand_aliases=["ciplar", "inderal"],
        soundalike_warnings=[],
    ),
    "furosemide": MedicineInfo(
        standard_name="Furosemide",
        category="Loop Diuretic",
        common_formulations=["TAB", "INJ"],
        common_dosages=["20 MG", "40 MG", "80 MG"],
        brand_aliases=["lasix", "frusenex"],
        soundalike_warnings=[],
    ),

    # Additional Antidiabetics
    "sitagliptin": MedicineInfo(
        standard_name="Sitagliptin",
        category="DPP-4 Inhibitor",
        common_formulations=["TAB"],
        common_dosages=["25 MG", "50 MG", "100 MG"],
        brand_aliases=["januvia", "istavel"],
        soundalike_warnings=[],
    ),
    "empagliflozin": MedicineInfo(
        standard_name="Empagliflozin",
        category="SGLT2 Inhibitor",
        common_formulations=["TAB"],
        common_dosages=["10 MG", "25 MG"],
        brand_aliases=["jardiance", "gibtulio"],
        soundalike_warnings=[],
    ),

    # Pain & Anti-inflammatory additions
    "mefenamic acid": MedicineInfo(
        standard_name="Mefenamic Acid",
        category="NSAID",
        common_formulations=["TAB", "SYP"],
        common_dosages=["250 MG", "500 MG", "50 MG/5ML"],
        brand_aliases=["meftal", "meftal-spas", "ponstan"],
        soundalike_warnings=[],
    ),
    "etoricoxib": MedicineInfo(
        standard_name="Etoricoxib",
        category="COX-2 Inhibitor NSAID",
        common_formulations=["TAB"],
        common_dosages=["60 MG", "90 MG", "120 MG"],
        brand_aliases=["nucoxia", "etoshine", "etody"],
        soundalike_warnings=[],
    ),

    # Vitamins & Supplements
    "calcium carbonate": MedicineInfo(
        standard_name="Calcium Carbonate",
        category="Mineral Supplement",
        common_formulations=["TAB", "SYP"],
        common_dosages=["500 MG", "1250 MG"],
        brand_aliases=["shelcal", "gemcal", "cipcal"],
        soundalike_warnings=[],
    ),
    "methylcobalamin": MedicineInfo(
        standard_name="Methylcobalamin",
        category="Vitamin B12",
        common_formulations=["TAB", "INJ"],
        common_dosages=["500 MCG", "1500 MCG"],
        brand_aliases=["neurobion", "nurokind", "mecobal"],
        soundalike_warnings=[],
    ),
}

# Standard formulation abbreviations and tokens
FORMULATIONS = {
    "tab": "TAB.",
    "tablet": "TAB.",
    "tabs": "TAB.",
    "cap": "CAP.",
    "capsule": "CAP.",
    "caps": "CAP.",
    "syp": "SYP.",
    "syrup": "SYP.",
    "inj": "INJ.",
    "injection": "INJ.",
    "oint": "OINT.",
    "ointment": "OINT.",
    "drops": "DROPS",
    "drop": "DROPS",
    "inh": "INH.",
    "inhaler": "INH.",
    "gel": "GEL",
    "susp": "SUSP.",
    "suspension": "SUSP.",
}

# Fast lookup dictionary mapping aliases, brands and lowercase standard names to standard names
ALIAS_TO_STANDARD: Dict[str, str] = {}
for generic_key, info in PHARMACOPEIA.items():
    ALIAS_TO_STANDARD[generic_key.lower()] = info.standard_name
    ALIAS_TO_STANDARD[info.standard_name.lower()] = info.standard_name
    for alias in info.brand_aliases:
        ALIAS_TO_STANDARD[alias.lower()] = info.standard_name
