// Host raster/label checks only. No neural computation and no GPU discovery.
#include "../ocean/sprite_forage/sprite_forage.h"
#include "forage_annotations.h"

int main(void) {
    if(forage_annotation_checks())return 1;
    Env env={0};unsigned char labels[FORAGE_AREA];env.player_idx=17;
    for(int cell=0;cell<FORAGE_AREA;cell++)env.grid[cell]=cell%FORAGE_TILE_TYPES;
    forage_visible_labels(&env,labels);
    for(int cell=0;cell<FORAGE_AREA;cell++)if(labels[cell]!=(cell==17?FORAGE_PLAYER:env.grid[cell]))return 2;
    // Full pixel frames from real puf_reset/step, with no changes to the env.
    Ini ini={0};puf_ini_load_file(&ini,"config/default.ini");puf_ini_load_file(&ini,"config/sprite_forage.ini");
    Dict* kwargs=puf_ini_section(&ini,"env",0);dict_set(kwargs,"observation_mode",1);
    float* pixels=calloc(FORAGE_PIXEL_OBS_SIZE,sizeof(float));float action=0,reward=0,terminal=0;
    unsigned char expected_labels[12][FORAGE_AREA];unsigned int expected_rng[12];int expected_player[12];
    for(int skin=0;skin<FORAGE_REPRESENTATIONS;skin++) {
        Env game={0};game.rng=7;game.agents[0].observations=pixels;game.agents[0].actions=&action;
        game.agents[0].rewards=&reward;game.agents[0].terminals=&terminal;
        dict_set(kwargs,"representation",skin);puf_init(&game,kwargs);puf_reset(&game);
        for(int step=0;step<12;step++) {
            forage_visible_labels(&game,labels);
            if(!skin) {
                memcpy(expected_labels[step],labels,FORAGE_AREA);expected_rng[step]=game.rng;expected_player[step]=game.player_idx;
            } else if(memcmp(expected_labels[step],labels,FORAGE_AREA)
                ||expected_rng[step]!=game.rng||expected_player[step]!=game.player_idx)return 4;
            for(int cell=0;cell<FORAGE_AREA;cell++)for(int y=0;y<FORAGE_CELL;y++) {
                int at=((cell/FORAGE_WIDTH*FORAGE_CELL+y)*OBS_WIDTH+cell%FORAGE_WIDTH*FORAGE_CELL)*3;
                if(memcmp(pixels+at,forage_tiles[skin][labels[cell]]+y*FORAGE_CELL*3,FORAGE_CELL*3*sizeof(float)))return 3;
            }
            action=step%5;puf_step(&game);
        }
        puf_close(&game);
    }
    free(pixels);puf_ini_free(&ini);puts("Forage: five skins, visible labels, native pixels and steps pass");return 0;
}
