#pragma once

// Original 16x16 indexed pixel art. No external images or game assets.
#define LD_SPRITE_SIZE 16
#define LD_TILE_TYPES 6
enum { LD_GRASS, LD_ROAD, LD_DIVIDER, LD_HAZARD, LD_COIN, LD_PLAYER };

static const char* const LD_NAMES[LD_TILE_TYPES] = {
    "grass", "road", "divider", "hazard", "coin", "player"
};

static const char* const LD_ART[LD_TILE_TYPES][16] = {
    { // Grass: flat ground beside the road.
        "................", "................", "................", "................",
        "................", "................", "................", "................",
        "................", "................", "................", "................",
        "................", "................", "................", "................"
    },
    { // Road: flat asphalt.
        "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa"
    },
    { // Lane edge: asphalt with a dashed white stripe on the left edge.
        "waaaaaaaaaaaaaaa", "waaaaaaaaaaaaaaa", "waaaaaaaaaaaaaaa", "waaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa",
        "waaaaaaaaaaaaaaa", "waaaaaaaaaaaaaaa", "waaaaaaaaaaaaaaa", "waaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa"
    },
    { // Hazard: red oncoming car, white windshield, black wheels.
        "....DDDDDDDD....", "....DRRRRRRD....", "...kRRRRRRRRk...", "...RRRRRRRRRR...",
        "...RRRRRRRRRR...", "...RRRwwRRRRR...", "...RRRwwRRRRR...", "...RRRRRRRRRR...",
        "...RRRRRRRRRR...", "...RRRRRRRRRR...", "...RRRRRRRRRR...", "...RRRRRRRRRR...",
        "...kRRRRRRRRk...", "....DRRRRRRD....", "....DDDDDDDD....", "................"
    },
    { // Coin: gold disc with a darker rim.
        "......YYYY......", ".....YYYYYY.....", "....YOOOOOYY....", "...YYOOOOOYYY...",
        "...YOYYYYYYOY...", "...YOYYYYYYOY...", "...YOYYYYYYOY...", "...YOYYYYYYOY...",
        "...YOYYYYYYOY...", "...YOYYYYYYOY...", "...YYOOOOOYYY...", "....YOOOOOYY....",
        ".....YYYYYY.....", "......YYYY......", "......YYYY......", "................"
    },
    { // Player: blue car, same silhouette as the hazard.
        "....BBBBBBBB....", "....BbbbbbbB....", "...kbbbbbbbbk...", "...bbbbbbbbbb...",
        "...bbbbbbbbbb...", "...bbbwwbbbbb...", "...bbbwwbbbbb...", "...bbbbbbbbbb...",
        "...bbbbbbbbbb...", "...bbbbbbbbbb...", "...bbbbbbbbbb...", "...bbbbbbbbbb...",
        "...kbbbbbbbbk...", "....BbbbbbbB....", "....BBBBBBBB....", "................"
    }
};
