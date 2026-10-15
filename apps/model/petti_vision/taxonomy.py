"""Label taxonomy shared by zero-shot prompts, trainable heads, matcher and UI.

Every label has a stable `key` (used in DB / training data), an English name
(used to build CLIP/SigLIP prompts) and a Mongolian name (shown to users).

Adding a breed/colour/mark here is enough for zero-shot to start predicting it;
the trainable heads pick it up next time you run `training/train_heads.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Label:
    key: str
    en: str
    mn: str
    # extra prompt phrasings (prompt ensembling improves zero-shot accuracy)
    alt: tuple[str, ...] = ()


@dataclass(frozen=True)
class Breed(Label):
    species: str = "dog"
    size: str = "medium"  # typical adult size -> prior for the size head


# --------------------------------------------------------------------------- #
# Species
# --------------------------------------------------------------------------- #
SPECIES: list[Label] = [
    Label("dog", "dog", "Нохой", ("puppy", "canine")),
    Label("cat", "cat", "Муур", ("kitten", "domestic cat")),
    Label("bird", "pet bird", "Шувуу", ("parrot", "budgie")),
    Label("rabbit", "rabbit", "Туулай", ("bunny",)),
    Label("rodent", "hamster or guinea pig", "Мэрэгч", ("hamster", "guinea pig")),
    Label("horse", "horse", "Морь", ()),
    Label("other", "other animal", "Бусад", ()),
]

# COCO class id (YOLO) -> species key
COCO_TO_SPECIES = {14: "bird", 15: "cat", 16: "dog", 17: "horse"}

# --------------------------------------------------------------------------- #
# Breeds. `size` is the typical adult size, used as a prior.
# --------------------------------------------------------------------------- #
_D = "dog"
_C = "cat"
BREEDS: list[Breed] = [
    # --- Dogs common in Mongolia first ---
    Breed("mixed_dog", "mixed breed dog", "Эрлийз нохой", ("mutt", "street dog"), _D, "medium"),
    Breed("bankhar", "Mongolian Bankhar dog", "Монгол банхар", ("large fluffy black and tan Mongolian herding dog",), _D, "large"),
    Breed("tibetan_mastiff", "Tibetan Mastiff", "Төвд мастиф", (), _D, "giant"),
    Breed("central_asian_shepherd", "Central Asian Shepherd dog", "Алабай", ("Alabai",), _D, "giant"),
    Breed("caucasian_shepherd", "Caucasian Shepherd dog", "Кавказ овчарк", (), _D, "giant"),
    Breed("german_shepherd", "German Shepherd", "Герман овчарк", (), _D, "large"),
    Breed("east_european_shepherd", "East European Shepherd", "Зүүн европ овчарк", (), _D, "large"),
    Breed("belgian_malinois", "Belgian Malinois", "Малинуа", (), _D, "large"),
    Breed("siberian_husky", "Siberian Husky", "Хаски", (), _D, "medium"),
    Breed("alaskan_malamute", "Alaskan Malamute", "Маламут", (), _D, "large"),
    Breed("samoyed", "Samoyed", "Самоед", (), _D, "medium"),
    Breed("golden_retriever", "Golden Retriever", "Голден ретривер", (), _D, "large"),
    Breed("labrador_retriever", "Labrador Retriever", "Лабрадор", (), _D, "large"),
    Breed("rottweiler", "Rottweiler", "Ротвейлер", (), _D, "large"),
    Breed("doberman", "Doberman Pinscher", "Доберман", (), _D, "large"),
    Breed("boxer", "Boxer dog", "Боксер", (), _D, "large"),
    Breed("great_dane", "Great Dane", "Дог", (), _D, "giant"),
    Breed("saint_bernard", "Saint Bernard", "Сенбернар", (), _D, "giant"),
    Breed("bernese_mountain", "Bernese Mountain Dog", "Бернийн уулын нохой", (), _D, "giant"),
    Breed("akita", "Akita", "Акита", (), _D, "large"),
    Breed("shiba_inu", "Shiba Inu", "Шиба ину", (), _D, "small"),
    Breed("chow_chow", "Chow Chow", "Чау чау", (), _D, "medium"),
    Breed("shar_pei", "Shar Pei", "Шарпей", (), _D, "medium"),
    Breed("border_collie", "Border Collie", "Бордер колли", (), _D, "medium"),
    Breed("australian_shepherd", "Australian Shepherd", "Австрали овчарк", (), _D, "medium"),
    Breed("shetland_sheepdog", "Shetland Sheepdog", "Шелти", (), _D, "small"),
    Breed("welsh_corgi", "Welsh Corgi", "Корги", (), _D, "small"),
    Breed("beagle", "Beagle", "Бигль", (), _D, "small"),
    Breed("cocker_spaniel", "Cocker Spaniel", "Кокер спаниель", (), _D, "small"),
    Breed("dalmatian", "Dalmatian", "Далматин", (), _D, "large"),
    Breed("pit_bull", "American Pit Bull Terrier", "Питбуль", ("Staffordshire terrier",), _D, "medium"),
    Breed("bulldog", "English Bulldog", "Бульдог", (), _D, "medium"),
    Breed("french_bulldog", "French Bulldog", "Франц бульдог", (), _D, "small"),
    Breed("pug", "Pug", "Мопс", (), _D, "small"),
    Breed("dachshund", "Dachshund", "Такс", (), _D, "small"),
    Breed("jack_russell", "Jack Russell Terrier", "Жэк рассел", (), _D, "small"),
    Breed("miniature_schnauzer", "Miniature Schnauzer", "Шнауцер", (), _D, "small"),
    Breed("pomeranian", "Pomeranian", "Померан", ("Pomeranian spitz",), _D, "toy"),
    Breed("spitz", "German Spitz", "Шпиц", (), _D, "small"),
    Breed("toy_poodle", "Toy Poodle", "Пудель", ("poodle",), _D, "toy"),
    Breed("maltese", "Maltese dog", "Мальтез", (), _D, "toy"),
    Breed("bichon_frise", "Bichon Frise", "Бишон фризе", (), _D, "small"),
    Breed("shih_tzu", "Shih Tzu", "Ши тзү", (), _D, "toy"),
    Breed("pekingese", "Pekingese", "Пекин нохой", (), _D, "toy"),
    Breed("yorkshire_terrier", "Yorkshire Terrier", "Йорк", (), _D, "toy"),
    Breed("chihuahua", "Chihuahua", "Чихуахуа", (), _D, "toy"),
    Breed("miniature_pinscher", "Miniature Pinscher", "Той пинчер", ("toy terrier",), _D, "toy"),
    Breed("papillon", "Papillon", "Папийон", (), _D, "toy"),
    Breed("cavalier_king_charles", "Cavalier King Charles Spaniel", "Кавалер кинг чарльз", (), _D, "small"),
    Breed("weimaraner", "Weimaraner", "Веймаранер", (), _D, "large"),
    # --- Cats ---
    Breed("domestic_shorthair", "domestic short-haired cat", "Энгийн богино үстэй муур", ("mixed breed cat",), _C, "medium"),
    Breed("domestic_longhair", "domestic long-haired cat", "Энгийн урт үстэй муур", (), _C, "medium"),
    Breed("british_shorthair", "British Shorthair cat", "Британи богино үст", (), _C, "medium"),
    Breed("scottish_fold", "Scottish Fold cat", "Шотланд вислоухий", ("cat with folded ears",), _C, "medium"),
    Breed("persian", "Persian cat", "Перс муур", (), _C, "medium"),
    Breed("exotic_shorthair", "Exotic Shorthair cat", "Экзотик", (), _C, "medium"),
    Breed("siamese", "Siamese cat", "Сиам муур", (), _C, "small"),
    Breed("thai", "Thai cat", "Тай муур", (), _C, "small"),
    Breed("maine_coon", "Maine Coon cat", "Мэйн кун", (), _C, "large"),
    Breed("ragdoll", "Ragdoll cat", "Рэгдолл", (), _C, "large"),
    Breed("siberian_cat", "Siberian cat", "Сибирь муур", (), _C, "large"),
    Breed("norwegian_forest", "Norwegian Forest cat", "Норвеги ойн муур", (), _C, "large"),
    Breed("bengal", "Bengal cat", "Бенгал муур", (), _C, "medium"),
    Breed("sphynx", "Sphynx hairless cat", "Сфинкс", (), _C, "medium"),
    Breed("russian_blue", "Russian Blue cat", "Орос цэнхэр", (), _C, "medium"),
    Breed("abyssinian", "Abyssinian cat", "Абиссин", (), _C, "small"),
    Breed("birman", "Birman cat", "Бирм муур", (), _C, "medium"),
    Breed("turkish_angora", "Turkish Angora cat", "Турк ангор", (), _C, "medium"),
    Breed("munchkin", "Munchkin cat", "Манчкин", ("short-legged cat",), _C, "small"),
    Breed("devon_rex", "Devon Rex cat", "Девон рекс", (), _C, "small"),
    Breed("bombay", "Bombay cat", "Бомбей", ("black cat",), _C, "medium"),
    Breed("egyptian_mau", "Egyptian Mau cat", "Египет мау", (), _C, "medium"),
]

BREED_BY_KEY = {b.key: b for b in BREEDS}


def breeds_for(species: str) -> list[Breed]:
    return [b for b in BREEDS if b.species == species]


# --------------------------------------------------------------------------- #
# Size / age
# --------------------------------------------------------------------------- #
SIZES: list[Label] = [
    Label("toy", "tiny toy-sized", "Маш жижиг (<4кг)"),
    Label("small", "small", "Жижиг (4-10кг)"),
    Label("medium", "medium-sized", "Дунд (10-25кг)"),
    Label("large", "large", "Том (25-45кг)"),
    Label("giant", "giant very large", "Аварга (45кг+)"),
]
SIZE_INDEX = {s.key: i for i, s in enumerate(SIZES)}

AGES: list[Label] = [
    Label("baby", "baby puppy or kitten", "Гөлөг / зулзага (<6 сар)", ("very young puppy", "very young kitten")),
    Label("young", "young juvenile", "Залуу (6 сар - 2 нас)", ("adolescent",)),
    Label("adult", "adult", "Насанд хүрсэн (2-7 нас)", ("grown-up",)),
    Label("senior", "old senior", "Өндөр настай (7+ нас)", ("elderly with grey muzzle",)),
]
AGE_INDEX = {a.key: i for i, a in enumerate(AGES)}

# --------------------------------------------------------------------------- #
# Colour & coat pattern
# --------------------------------------------------------------------------- #
COLORS: list[Label] = [
    Label("black", "black", "Хар"),
    Label("white", "white", "Цагаан"),
    Label("grey", "grey", "Саарал", ("blue grey",)),
    Label("brown", "brown", "Бор", ("chocolate brown",)),
    Label("tan", "tan", "Шаргал хүрэн", ("fawn",)),
    Label("golden", "golden", "Алтан шар", ("yellow",)),
    Label("red", "red ginger orange", "Улаан шар", ("ginger", "orange")),
    Label("cream", "cream", "Цөцгий өнгө", ("beige",)),
    Label("silver", "silver", "Мөнгөлөг"),
]
# reference RGB for the pixel-based colour estimator (see color.py)
COLOR_RGB = {
    "black": (25, 25, 25),
    "white": (235, 235, 230),
    "grey": (128, 128, 130),
    "brown": (100, 60, 35),
    "tan": (170, 125, 80),
    "golden": (210, 165, 80),
    "red": (190, 95, 40),
    "cream": (225, 205, 165),
    "silver": (180, 180, 185),
}

PATTERNS: list[Label] = [
    Label("solid", "solid single-coloured coat", "Нэг өнгө"),
    Label("bicolor", "two-coloured coat", "Хоёр өнгө", ("black and white", "tuxedo")),
    Label("tricolor", "tri-colour coat", "Гурван өнгө", ("calico",)),
    Label("tabby", "tabby striped coat", "Судалтай (табби)", ("striped",)),
    Label("spotted", "spotted coat", "Толботой", ("dalmatian spots",)),
    Label("brindle", "brindle coat", "Бриндл (барс судал)", ()),
    Label("merle", "merle coat", "Мерле", ()),
    Label("tortoiseshell", "tortoiseshell coat", "Яст мэлхий (тортойз)", ()),
    Label("pointed", "colour-point coat with dark face ears and paws", "Колор-поинт", ("siamese pattern",)),
    Label("black_and_tan", "black and tan coat with tan eyebrows", "Хар хүрэн (4 нүдтэй)", ("dog with tan spots above eyes",)),
]

COAT_LENGTH: list[Label] = [
    Label("hairless", "hairless", "Үсгүй"),
    Label("short", "short-haired", "Богино үстэй"),
    Label("medium", "medium-length fur", "Дунд урт үстэй"),
    Label("long", "long fluffy fur", "Урт сэвсгэр үстэй"),
    Label("curly", "curly fur", "Буржгар үстэй"),
]

# --------------------------------------------------------------------------- #
# Distinctive marks (multi-label, each is a binary positive/negative prompt pair)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mark(Label):
    negative: str = ""
    species: tuple[str, ...] = ("dog", "cat")
    threshold: float = 0.65  # probability needed to report it
    weight: float = 1.0  # how discriminative it is for matching
    tags: tuple[str, ...] = field(default=())


MARKS: list[Mark] = [
    Mark("collar", "wearing a collar", "Хүзүүвчтэй", negative="without a collar", weight=0.4),
    Mark("harness", "wearing a harness", "Оосортой (harness)", negative="without a harness", weight=0.3),
    Mark("clothes", "wearing clothes", "Хувцастай", negative="not wearing clothes", weight=0.3),
    Mark("ear_tip", "with a clipped ear tip", "Чихний үзүүр тайрсан (TNR)", negative="with intact ears", weight=1.5, threshold=0.8),
    Mark("heterochromia", "with two different coloured eyes", "Хоёр өөр өнгийн нүдтэй", negative="with matching eye colours", weight=2.0, threshold=0.8),
    Mark("blue_eyes", "with blue eyes", "Цэнхэр нүдтэй", negative="with brown or yellow eyes", weight=1.0),
    Mark("one_eye", "with one eye missing", "Нэг нүдгүй", negative="with two eyes", weight=2.5, threshold=0.85),
    Mark("short_tail", "with a docked or very short tail", "Сүүл богино/тайрсан", negative="with a long tail", weight=1.2),
    Mark("three_legs", "with three legs, missing a leg", "Гурван хөлтэй", negative="with four legs", weight=2.5, threshold=0.85),
    Mark("scar", "with a visible scar or wound", "Сорви/шархтай", negative="with no scars", weight=1.5, threshold=0.8),
    Mark("white_chest", "with a white patch on the chest", "Цээжиндээ цагаан толботой", negative="with no white on the chest", weight=1.0),
    Mark("white_paws", "with white paws like socks", "Цагаан сарвуутай (оймс)", negative="with no white on the paws", weight=1.0),
    Mark("face_mask", "with a dark mask on the face", "Нүүрэндээ бараан масктай", negative="with no dark face mask", weight=0.8),
    Mark("eye_patch", "with a patch of colour around one eye", "Нэг нүдэндээ толботой", negative="with an evenly coloured face", weight=1.5),
    Mark("blaze", "with a white stripe down the middle of the face", "Магнайдаа цагаан зураастай", negative="with no white stripe on the face", weight=1.2),
    Mark("floppy_ears", "with floppy ears", "Унжсан чихтэй", negative="with erect pointed ears", weight=0.6),
    Mark("folded_ears", "with folded ears", "Нугалсан чихтэй", negative="with upright ears", species=("cat",), weight=1.0),
    Mark("curled_tail", "with a tail curled over its back", "Сүүлээ дээш муруйлгасан", negative="with a straight hanging tail", species=("dog",), weight=0.6),
    Mark("four_eyes", "with tan spots above the eyes", "Дөрвөн нүдтэй (хөмсөг толбо)", negative="with no spots above the eyes", species=("dog",), weight=1.0),
]
MARK_BY_KEY = {m.key: m for m in MARKS}


TASKS = {
    "species": SPECIES,
    "breed": BREEDS,
    "size": SIZES,
    "age": AGES,
    "color": COLORS,
    "pattern": PATTERNS,
    "coat": COAT_LENGTH,
}
MULTI_LABEL_TASKS = {"color", "marks"}


def label_by_key(task: str, key: str) -> Label:
    for lab in TASKS[task] if task != "marks" else MARKS:
        if lab.key == key:
            return lab
    raise KeyError(f"{task}:{key}")
