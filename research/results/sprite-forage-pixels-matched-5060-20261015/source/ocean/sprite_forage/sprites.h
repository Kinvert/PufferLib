#pragma once

// Original 16x16 indexed pixel art. No external images or game assets.
// The two negative lookalikes reuse silhouettes AND color counts; their
// distinguishing information is where the highlight/texture pixels occur.
#define FORAGE_SPRITE_SIZE 16
#define FORAGE_TILE_TYPES 13
enum { FORAGE_EMPTY, FORAGE_APPLE, FORAGE_PEAR, FORAGE_GRAPE,
    FORAGE_LEAF, FORAGE_FLOWER, FORAGE_BEETLE, FORAGE_MUSHROOM,
    FORAGE_SPOTTED_APPLE, FORAGE_WASP, FORAGE_TOADSTOOL,
    FORAGE_TREE, FORAGE_PLAYER };

static const char* const FORAGE_NAMES[FORAGE_TILE_TYPES] = {
    "ground", "apple", "pear", "grapes", "leaf", "flower", "beetle",
    "striped mushroom", "spotted apple", "wasp", "spotted toadstool", "tree", "explorer"
};

static const char* const FORAGE_ART[FORAGE_TILE_TYPES][16] = {
    {0},
    { // Apple: stem, folded leaf, outlined fruit and a continuous shine.
        "................", "........ss......", "......gggs......", ".....gGGs.......",
        "....kkk.kkk.....", "...krrrkrrrk....", "..kroorrrrrrk...", ".krooorrrrrrrk..",
        ".kroorrrrrrrrk..", ".krrrrrrrrrrrk..", ".krrrrrrrrrrrk..", "..krrrrrrrrrk...",
        "..krrrrrrrrrk...", "...krrrrrrrk....", "....kkkrkkk.....", "................"
    },
    { // Pear: narrow neck and asymmetric curved belly.
        "........s.......", ".......sgg......", "......kgggk.....", "......kyyk......",
        ".....kyyyyk.....", ".....kyoyyk.....", "....kyoyyyyk....", "...kyoyyyyyyk...",
        "..kyoyyyyyyyyk..", "..kyoyyyyyyyyk..", "..kyyyyyyyyyyk..", "..kyyyyyyyyyyk..",
        "...kyyyyyyyyk...", "....kyyyyyyk....", ".....kkkkkk.....", "................"
    },
    { // A cluster of berries, individual round lobes and a twig.
        "........s.......", "......ggs.......", ".....gGGg.......", "....kkk.kkk.....",
        "...kpwpkpwpk....", "...kpppkpppk....", "....kkkkkkk.....", "..kkk.kkk.kkk...",
        ".kpwpkpwpkpwpk..", ".kpppkpppkpppk..", "..kkkkkkkkkkk...", "....kkk.kkk.....",
        "...kpwpkpwpk....", "...kpppkpppk....", "....kkkkkkk.....", "................"
    },
    { // Leaf: serrated silhouette, central vein and diagonal side veins.
        ".............g..", "..........gggk..", "........gggggk..", "......gggygggk..",
        ".....gggygggk...", "....gggygyggk...", "...gggyggggk....", "..gggygygggk....",
        "..ggygggggk.....", "..gygygggk......", "..gyggggk.......", "..gygggk........",
        "...gygk.........", "...syk..........", "..ss............", "................"
    },
    { // Flower: separate petals, a center, stalk and paired leaves.
        "......kk........", ".....kppk.......", "...kkkppkkk.....", "..kppkyykppk....",
        "..kppyyyyppk....", "...kkyyyykk.....", "..kppkyykppk....", "..kppkkkkppk....",
        "...kk.gg.kk.....", "......gg........", "...gggggggg.....", "....GGggGG......",
        "......gg........", "......gg........", ".....kggk.......", "................"
    },
    { // Beetle: antennae, segmented carapace, eyes and six legs.
        "....k......k....", ".....k....k.....", ".....kkkkkk.....", "....kbwwwwbk....",
        "....kbbkkbbk....", "..kkkbbkkbbkkk..", "....kbbyybbk....", "...kbbbyybbbk...",
        ".kkkbbbkkbbbkkk.", "...kbbbkkbbbk...", "....kbbyybbk....", "..kkkbbkkbbkkk..",
        "....kbbbbbbk....", ".....kkkkkk.....", "....kk....kk....", "................"
    },
    { // Mushroom: banded cap, gills, stalk and feet.
        "................", "......kkkk......", "....kkrrorkk....", "...krrroorrrk...",
        "..krrroorrrrrk..", ".krrroorrrrrrrk.", ".krrroorrrrrrrk.", ".krrroorrrrrrrk.",
        "..kkkkkkkkkkkk..", "...kssssssssk...", ".....kwwwwk.....", ".....kwnwwk.....",
        ".....kwwnwk.....", "....kwwwwwwk....", "....kkkkkkkk....", "................"
    },
    { // Spotted apple: same silhouette/color counts, scattered highlights.
        "................", "........ss......", "......gggs......", ".....gGGs.......",
        "....kkk.kkk.....", "...krrrkrrrk....", "..krrrrrrrrrk...", ".krrrrrrrrrrrk..",
        ".krrrrrrrrrrrk..", ".krrrrrorrrrrk..", ".krorrrrrrorrk..", "..krrrrorrrrk...",
        "..krrorrrrrrk...", "...korrrrrrk....", "....kkkokkk.....", "................"
    },
    { // Wasp: translucent-looking wings, eyes, banded abdomen and stinger.
        "................", "...kk......kk...", "..kwwk....kwwk..", ".kwwwwk..kwwwwk.",
        "..kwwwwkkwwwwk..", "...kkkkkkkkkk...", "....kyykkkkyk...", "...kyyykkkyyyk..",
        "..kyyyykkkyyykk.", "..kyyyykkkyyyk..", "...kyyykkkyyk...", "....kkkkkkkk....",
        ".....k.k.k......", "....k..k..k.....", "..........k.....", "................"
    },
    { // Spotted toadstool: same cap silhouette/palette, scattered markings.
        "................", "......kkkk......", "....kkrorrkk....", "...krrorrrrok...",
        "..krrrrorrrrok..", ".krrrrrrrorrrrk.", ".korrrrorrrrork.", ".krrrorrrrorrrk.",
        "..kkkkkkkkkkkk..", "...kssssssssk...", ".....kwwwwk.....", ".....kwnwwk.....",
        ".....kwwnwk.....", "....kwwwwwwk....", "....kkkkkkkk....", "................"
    },
    { // A tree is blocking scenery, not a collectible.
        "......kkkk......", "....kkggggkk....", "...kgggGGgggk...", "..kggGGggggggk..",
        ".kggggggGGggggk.", ".kggGGggggggggk.", "kggggggGGggGGggk", "kggGGggggggggggk",
        ".kggggggggGGggk.", "..kkkkgGGgkkkk..", ".....kssssk.....", ".....kssssk.....",
        ".....kssssk.....", "....kssssssk....", "...kssssssssk...", "....kkkkkkkk...."
    },
    { // Original explorer: cap, face, arms, jacket, belt and boots.
        ".....kkkkkk.....", "....kggggggk....", "...kggggggggk...", "....knnnnnnk....",
        "....knknnknk....", "....knnnnnnk....", ".....knnnnk.....", "...kkkbbbbkkk...",
        "..knnkbbbbknnk..", "..knnkbbbbknnk..", "...kkbyyyykkk...", ".....kbbbbk.....",
        ".....kbkkbk.....", "....kbbkkbbk....", "...ksss..sssk...", "...kkkk..kkkk..."
    }
};
