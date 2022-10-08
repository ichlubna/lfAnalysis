// 1 warp načte 32 px
// matce rows x cols A = 32*16 + B = 16*8 = C = 32*8
// A 8 pixelů z 16 pohledů
// B 16 vah pro 8 snímků
// warp 4x (pro všech 32 - 4*8) udělá  AB + AB + AB + AB = 8 pixelů pro 8 snímků
// 4 váhovací matice po 16 pohledech a 8 snímcích, 4 matice po 8 pxelech pro 16 pohledů
// váhovací matice zůstávají takže jen 4*4 matic po 8 pxelech
// pokud 32 pixelů a načítat po 16 views 2* 32*16 *4 * 8 + 8*32*2*4*8 rtx 4090
// pokud po 16 pixelech 2* 32*16 *2 *8 + 8*32*2*2*8
pro začátek pon jednom, načíst 8 pixelů
