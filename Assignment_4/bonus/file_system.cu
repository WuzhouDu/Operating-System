#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

__device__ __managed__ int current_dir_index = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;
  

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  
    // init volume FCB
  fs->volume[fs->SUPERBLOCK_SIZE] = '/';
  fs->volume[fs->SUPERBLOCK_SIZE + 1] = '\0';
  fs->volume[fs->SUPERBLOCK_SIZE + 28] = 1;
  fs->volume[fs->SUPERBLOCK_SIZE + 29] = 1;
  fs->volume[fs->SUPERBLOCK_SIZE + 30] = 0xff;
  fs->volume[fs->SUPERBLOCK_SIZE + 31] = 0xff;

  // init volume VCB
  fs->volume[0] = 0b10000000;
}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */

  /* Searching the file name in FCB */
  int found = 0;
  int current_depth = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 28];
  u32 fp;
  

  for (int i = 0; i < fs->FCB_ENTRIES; i++){
    if (!found){
      if (((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31]) != current_dir_index) continue;
      for (int j = 0; j < fs->FCB_SIZE; j++){
        if (s[j] != fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) break;
        else if ((s[j] == fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) && (s[j] == '\0')) {
          // if found the file, return
          found = 1;
          fp = fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 20]) << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 21]);
          // printf("[fs_open] : Open existing file: %s, fp is %x, FCB = %d, start block index = %x, length = %d, time = %d\n", 
          //   s, fp, i,
          //   (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 21],
          //   (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 23],
          //   (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 24] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 25]);
          return fp;
        }
      }
    }
  }

  if ((!found) && (op == G_READ)) {
    printf("[fs_open ERROR] : G_READ nonexisting file: %s. RETURN.\n", s);
    return 0XFFFFFFFF;
  }
  if ((!found) && (op == G_WRITE)){
    int name_length;

    // printf("[fs_open] : G_WRITE new file: %s.\n", s);
    for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++){
      if (s[i] == '\0'){
        name_length = i + 1;
        break;
      }
    }
  
  /* Handle the G_WRITE new file. */

    /* traverse the VCB to record the first available block */
    int first_avai_block = 0xffff;
    for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++){
      if (fs->volume[i] != 0xff){
        for (int j = 7; j >= 0; j--){
          if (((fs->volume[i] & (1 << j)) >> j) == 0) {
            // record this block
            first_avai_block = i*8 + (7-j);

            // write to VCB to allocate memory.
            fs->volume[i] += (1 << j);
            // printf("[fs_open] : find available block %u\n", first_avai_block);
            // printf("[fs_open] : update VCB %d to %x\n", i, fs->volume[i]);
            break;
          }
        }
        break;
      }
    }
    if (first_avai_block == 0xffff) {
      printf("[ERROR fs_open] : no available block.\n");
      return 0XFFFFFFFF;
    }

    /* traverse the FCB to find first available FCB*/
    int first_avai_FCB = 0xfff;
    for (int i = 0; i < fs->FCB_ENTRIES; i++){
      if (first_avai_FCB == 0xfff){
        for (int j = 0; j < 20; j++){ // browse the file name of this FCB to judge whether available.
          if (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j] != 0) break;
          if (j == 19) {
            first_avai_FCB = i;
            // printf("[fs_open] : find available FCB %x\n", first_avai_FCB);
            break;
          }
        }
      }
      else break;
    }

    if (first_avai_FCB == 0xfff) {
      printf("[ERROR fs_open] : no available FCB.\n");
      return 0XFFFFFFFF;
    }
    
    /* write the meta data to this available FCB */
    // write the file name.
    for (int i = 0; i < 20; i++){
      fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + i] = s[i];
      if (s[i] == '\0') break;
    }
    
    // write the start point, length = 0, time, create time, depth, identity and parent.
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 20] = (first_avai_block >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 21] = (first_avai_block & 0xff);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 24] = (gtime >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 25] = (gtime & 0xff);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 26] = (gtime >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 27] = (gtime & 0xff);    
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 28] = current_depth + 1;
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 30] = (current_dir_index >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 31] = (current_dir_index & 0xff);

    fp = fs->FILE_BASE_ADDRESS + ((fs->STORAGE_BLOCK_SIZE)*(((fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 20]) << 8) + fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 21]));

    /* write the parent dir */
    int parent_dir_length = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23];
    int parent_dir_block_size = (parent_dir_length == 0) ? 1 : 1 + (parent_dir_length - 1) / fs->STORAGE_BLOCK_SIZE;
    int parent_dir_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 21];
    int new_parent_dir_length = parent_dir_length + name_length;
    int new_parent_dir_block_size = (new_parent_dir_length == 0) ? 1 : 1 + (new_parent_dir_length - 1) / fs->STORAGE_BLOCK_SIZE;
    int parent_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 31];
    char new_dir_contents[1024];
    char parent_dir_name[20];
    char parent_dir_invariant[4];

    if (parent_dir_block_size == new_parent_dir_block_size) {
      /* update FCB of parent dir */
        // length
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] = (new_parent_dir_length >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23] = (new_parent_dir_length & 0xff);
        // modification time
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

      /* update the contents of parent dir */
      for (int i = 0; i < name_length; i++){
        fs->volume[fs->FILE_BASE_ADDRESS + parent_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + parent_dir_length + i] = s[i];
      }
    }
    else{
      for (int i = 0; i < parent_dir_length; i++){
        new_dir_contents[i] = fs->volume[fs->FILE_BASE_ADDRESS + parent_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i];
      }
      for (int i = 0; i < name_length; i++){
        new_dir_contents[parent_dir_length + i] = s[i];
      }
      for (int i = 0; i < 20; i++){
        parent_dir_name[i] = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + i];
        if (parent_dir_name[i] == '\0') break;
      }
      for (int i = 0; i < 4; i++){
        parent_dir_invariant[i] = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 26 + i];
      }

      /* do the compaction */
      int new_avai_FCB = fs->MAX_FILE_NUM - 1;
      int new_avai_block = (1 << 15) - 1;

      // compact the contents of file
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = (parent_dir_start_block + i) / 8;
        int VCB_offset = (parent_dir_start_block + i) % 8;
        if ((fs->volume[VCB_index] & (1 << (7 - VCB_offset))) != 0){
          if (parent_dir_start_block + i + parent_dir_block_size >= (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE)){
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(parent_dir_start_block + i) + j] = 0;
            }
          }
          else {
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(parent_dir_start_block + i) + j] = fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(parent_dir_start_block + i +parent_dir_block_size) + j];
            }
          }
        }
        else {
          new_avai_block = parent_dir_start_block + i - parent_dir_block_size;
          break;
        }
      }
      
      // compact the VCB (from the backward direction)
      // find the last allocated block
      int last_block = (1 << 15);
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = i / 8;
        int VCB_offset = i % 8;
        if ((fs->volume[VCB_index] & (1 << (7-VCB_offset))) == 0) {
          last_block = i - 1;
          // printf("write compact: 最后一个block index为 %d\n", last_block);
          break;
        }
      }
      
      for (int i = 0; i < parent_dir_block_size; i++){
        int VCB_index = (last_block - i) / 8;
        int VCB_offset = (last_block - i) % 8;
        fs->volume[VCB_index] &= (0xff - (1 << (7 - VCB_offset)));
        // printf("压VCB: VCB[0], VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      }
      

      // compact the FCB
      for (int i = 0; i < fs->FCB_ENTRIES; i++){
        if (current_dir_index + i + 1 >= fs->FCB_ENTRIES) {
          for (int j = 0; j < fs->FCB_SIZE; j++){
            fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = 0;
          }
          break;
        }

        int vacant_next_FCB = 1;
        for (int j = 0; j < 20; j++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + j] != 0) {
            vacant_next_FCB = 0;
            break;
          }
        }
        if (vacant_next_FCB){
          for (int j = 0; j < fs->FCB_SIZE; j++){
            fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = 0;
          }
          new_avai_FCB = i;
          break;
        }

        int prev_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 21];
        int new_start_block = prev_start_block - parent_dir_block_size;
        
        // update name
        for (int j = 0; j < 20; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i + 1)*(fs->FCB_SIZE) + j];
        }

        // update start block index
        fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 20] = (new_start_block >> 8);
        fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 21] = (new_start_block & 0xff);
        
        // update length, modification time, create time, depth, identity
        for (int j = 22; j < 30; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + j];
        }

        // update second half parents
        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index == current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (new_avai_FCB >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (new_avai_FCB & 0xff);
        }
        else if (this_parent_index > current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
        }
        else{
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (this_parent_index >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (this_parent_index & 0xff);
        }
      }
      // update first half parents
      for (int i = 0; i < current_dir_index; i++){
        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index == current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = (new_avai_FCB >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = (new_avai_FCB & 0xff);
        }
        else if (this_parent_index > current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
        }
      }

      /* compaction finished */
      /* write new parent dir */

        // write to VCB
      int new_VCB_index;
      int new_VCB_offset;
      for (int i = 0; i < new_parent_dir_block_size; i++){
        new_VCB_index = (i + new_avai_block) / 8;
        new_VCB_offset = (i + new_avai_block) % 8;
        fs->volume[new_VCB_index] |= (1 << (7 - new_VCB_offset));
        // printf("写VCB: VCB[0], VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      }
      
      
      // write to FCB
      for (int i = 0; i < 20; i++) {
        fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + i] = parent_dir_name[i];
        if (parent_dir_name[i] == '\0') {
          break;
        }
      }
      
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 20] = (new_avai_block >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 21] = (new_avai_block & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 22] = (new_parent_dir_length >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 23] = (new_parent_dir_length & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 25] = (gtime & 0xff);
      for (int i = 0; i < 4; i++){
        fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 26 + i] = parent_dir_invariant[i];
      }
        // update parent in FCB
      if ((parent_parent_index > current_dir_index) && (parent_parent_index != 0xffff)) parent_parent_index--;
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 30] = (parent_parent_index >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 31] = (parent_parent_index & 0xff);
      
      // write to file content
      for (int i = 0; i < new_parent_dir_length; i++){
        fs->volume[fs->FILE_BASE_ADDRESS + new_avai_block*(fs->STORAGE_BLOCK_SIZE) + i] = new_dir_contents[i];
      }
      
      // update current dir index
      current_dir_index = new_avai_FCB;
    }

    // printf("[fs_open] : write meta data to FCB %x, name = %s, time = %x, length = 0, start address = %x\n", first_avai_FCB, s, gtime, fp);
    gtime++;
    return fp;
  }
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	for (int i = 0; i < size; i++){
    output[i] = fs->volume[fp + i];
    // printf("[fs_read] : read %x\n", output[i]);
   
  }
  // printf("[fs_read] : fs_read\n");
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	int block_index = (fp - fs->FILE_BASE_ADDRESS)/fs->STORAGE_BLOCK_SIZE;
  int write_block_size = (size == 0) ? 1 : 1 + (size-1) / fs->STORAGE_BLOCK_SIZE;
  // int write_block_size = size / fs->STORAGE_BLOCK_SIZE;
  int older_block_size = 0;
  int older_length = 0;
  // int older_create_time;
  int parent_FCB_index;
  int FCB_index = 0xfff;
  uchar name[20];
  uchar older_FCB_invariant[4];

  /* traverse the FCB to get the metadata of this file */
  for (int i = 0; i < fs->FCB_ENTRIES; i++){
    if (((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 21]) == block_index){
      FCB_index = i;
      older_length = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 23];
      older_block_size = (older_length == 0) ? 1 : 1 + (older_length-1) / fs->STORAGE_BLOCK_SIZE;
      // older_create_time = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 26] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 27];
      // older_block_size = older_length / fs->STORAGE_BLOCK_SIZE;
      parent_FCB_index = (fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 31];
      if (parent_FCB_index != current_dir_index){
        printf("[fs_write] : ERROR, not in the correct directory.\n");
        return;
      }

      for (int j = 0; j < 20; j++){
        name[j] = fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j];
        if (name[j] == '\0') break;
      }
      for (int j = 0; j < 4; j++){
        older_FCB_invariant[j] = fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 26 + j];
      }

      // printf("[fs_write] : find older file. FCB = %d, name = %s, start block index %x, length = %d, time = %d\n", 
        // FCB_index, name, block_index, 
        // (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 23],
        // (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 24] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 25]);
      break;
    }
  }
  if (FCB_index == 0xfff) {
    printf("[ERROR fs_write] : nonexist file in FCB\n");
    return 0xffffffff;
  }

  // if the original block size is the same as the new write block size, the block index will not change
  if (older_block_size == write_block_size){

    // rewrite the file content
    for (int i = 0; i < write_block_size*(fs->STORAGE_BLOCK_SIZE); i++){
      fs->volume[fs->FILE_BASE_ADDRESS + block_index*(fs->STORAGE_BLOCK_SIZE) + i] = (i < size) ? input[i] : 0;
    }

    // rewrite the FCB, only length and modification time need to rewrite
    fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 22] = (size >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 23] = (size & 0xff);
    fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

    // rewrite the parent directory's modification time
    fs->volume[fs->SUPERBLOCK_SIZE + parent_FCB_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + parent_FCB_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);
    // printf("[fs_write] : lucky dog! write file is the same block size as old file. write length = %d\n", size);
    // printf("暂时看VCB[0] VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
    gtime++;
    return 0;
  }

  /* else if original block size is different from the new write block size */
  else{
    // search next FCB's name to see whether this is the last file
    int last_file = 1;
    for (int i = 0; i < 20; i++){
      if (FCB_index == fs->FCB_ENTRIES - 1) break;
      if (last_file) {
        if (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + 1)*(fs->FCB_SIZE) + i] != 0){
          last_file = 0;
          break;
        }
      }
    }

    // if this file is the last file. no need to compact
    if (last_file || (FCB_index == fs->FCB_ENTRIES - 1)) {
      
      if (older_length <= size) {
        // rewrite the file content
        for (int i = 0; i < size; i++){
          fs->volume[fs->FILE_BASE_ADDRESS + block_index*(fs->STORAGE_BLOCK_SIZE) + i] = input[i];
        }
      }
      else {
        // rewrite the file content
        for (int i = 0; i < older_length; i++){
          fs->volume[fs->FILE_BASE_ADDRESS + block_index*(fs->STORAGE_BLOCK_SIZE) + i] = (i < size) ? input[i] : 0;
        }
      }
      
      // rewrite the FCB, only length and modification time need to rewrite
      fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 22] = (size >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 23] = (size & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

      // update VCB
      int VCB_index;
      int VCB_offset;
      for (int i = 0; i < write_block_size; i++){
        VCB_index = (i + block_index) / 8;
        VCB_offset = (i + block_index) % 8;
        fs->volume[VCB_index] |= (1 << (7 - VCB_offset));
        if (i == write_block_size - 1){
          if (write_block_size < older_block_size) {
            for (int j = 0; j < older_block_size - write_block_size; j++){
              VCB_index = (i + block_index+1 + j) / 8;
              VCB_offset = (i + block_index+1 + j) % 8;
              fs->volume[VCB_index] &= (0xff - (1 << (7 - VCB_offset))); 
            }
          }
        }
      }

      // update its parent dir FCB
      fs->volume[fs->SUPERBLOCK_SIZE + parent_FCB_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + parent_FCB_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

      // printf("[fs_write] : write file is the last file as before. write from block %d, length = %d\n", block_index, size);
      gtime++;
      return 0;
    }

    // block size different, and is not the last file. need to compact all the files after this file, then write to an available FCB
    
    /* do the compaction */
    else {
      int new_avai_FCB = fs->MAX_FILE_NUM - 1;
      int new_avai_block = (1 << 15) - 1;

      // compact the contents of file
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = (block_index + i) / 8;
        int VCB_offset = (block_index + i) % 8;
        if ((fs->volume[VCB_index] & (1 << (7 - VCB_offset))) != 0){
          if (block_index + i + older_block_size >= (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE)){
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(block_index + i) + j] = 0;
            }
          }
          else {
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(block_index + i) + j] = fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(block_index + i +older_block_size) + j];
            }
          }
        }
        else {
          new_avai_block = block_index + i - older_block_size;
          break;
        }
      }
      
      // compact the VCB (from the backward direction)
      // find the last allocated block
      int last_block = (1 << 15);
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = i / 8;
        int VCB_offset = i % 8;
        if ((fs->volume[VCB_index] & (1 << (7-VCB_offset))) == 0) {
          last_block = i - 1;
          // printf("write compact: 最后一个block index为 %d\n", last_block);
          break;
        }
      }
      
      for (int i = 0; i < older_block_size; i++){
        int VCB_index = (last_block - i) / 8;
        int VCB_offset = (last_block - i) % 8;
        fs->volume[VCB_index] &= (0xff - (1 << (7 - VCB_offset)));
        // printf("压VCB: VCB[0], VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      }
      
      // compact the FCB
      for (int i = 0; i < fs->FCB_ENTRIES; i++){
        if (FCB_index + i + 1 >= fs->FCB_ENTRIES) {
          for (int j = 0; j < fs->FCB_SIZE; j++){
            fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = 0;
          }
          break;
        }

        int vacant_next_FCB = 1;
        for (int j = 0; j < 20; j++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + j] != 0) {
            vacant_next_FCB = 0;
            break;
          }
        }
        if (vacant_next_FCB){
          for (int j = 0; j < fs->FCB_SIZE; j++){
            fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = 0;
          }
          new_avai_FCB = i;
          break;
        }        

        int prev_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 21];
        int new_start_block = prev_start_block - older_block_size;
        
        // update name
        for (int j = 0; j < 20; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i + 1)*(fs->FCB_SIZE) + j];
        }

        // update start block index
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 20] = (new_start_block >> 8);
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 21] = (new_start_block & 0xff);
        
        // update length, modification time, create time, depth, identity.
        for (int j = 22; j < 30; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + j];
        }

        // update second half parents
        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index > FCB_index){
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
        }
        else{
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 30] = (this_parent_index >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 31] = (this_parent_index & 0xff);
        }
      
      }
      // update first half parents
      for (int i = 0; i < FCB_index; i++){
        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index > FCB_index){
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
        }
      }

      // update the current dir
      if (current_dir_index > FCB_index) current_dir_index--;

      /* compaction finished. */

      // start write new file.
      // write to VCB
      int new_VCB_index;
      int new_VCB_offset;
      for (int i = 0; i < write_block_size; i++){
        new_VCB_index = (i + new_avai_block) / 8;
        new_VCB_offset = (i + new_avai_block) % 8;
        fs->volume[new_VCB_index] |= (1 << (7 - new_VCB_offset));
        // printf("写VCB: VCB[0], VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      }
      
      
      // write to FCB
      for (int i = 0; i < 20; i++) {
        // printf("name[%d] = %c\n", i, name[i]);
        // printf("8\n");
        // printf("new_avai_FCB = %x\n", new_avai_FCB);
        fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + i] = name[i];
        if (name[i] == '\0') {
          break;
        }
      }
      
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 20] = (new_avai_block >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 21] = (new_avai_block & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 22] = (size >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 23] = (size & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 25] = (gtime & 0xff);
      for (int i = 0; i < 4; i++){
        fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 26 + i] = older_FCB_invariant[i];
      }
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 30] = (current_dir_index >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 31] = (current_dir_index & 0xff);
      
      // write to file content
      for (int i = 0; i < size; i++){
        fs->volume[fs->FILE_BASE_ADDRESS + new_avai_block*(fs->STORAGE_BLOCK_SIZE) + i] = input[i];
      }

      // update the parent dir FCB
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

      // printf("[fs_write] : write between files. write from block %d, length = %d\n", new_avai_block, size);
      // printf("暂时看VCB[0] VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      gtime++;
      return 0;
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  if (op == LS_D){
    // sort by modified time
    int prev_max;
    int now_max;
    int now_max_index;
    
    printf("[fs_gsys] : ===sort by modified time===\n");
    for (int i = 0; i < fs->FCB_ENTRIES; i++){
      int vacant = 1;
      for (int n = 0; n < 20; n++){
        if (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + n] != 0){
          vacant = 0;
          // printf("没有空啊\n");
          break;
        }
      }
      if (vacant) return;
      int complete = 1;
      // find minimum time.
      if (i == 0) prev_max = 2147483647;
      for (int j = 0; j < fs->FCB_ENTRIES; j++){
        int vacant = 1;
        for (int n = 0; n < 20; n++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + n] != 0){
            vacant = 0;
            // printf("没有空啊\n");
            break;
          }
        }
        if (vacant) break;

        int time = (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 24] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 25];
        if (time < prev_max){
          now_max = time;
          now_max_index = j;
          complete = 0;
          // printf("这里time为%d, index为%d\n", now_max, now_max_index);
          break;
        }
      }
      if (complete) return;

      for (int j = 0; j < fs->FCB_ENTRIES; j++){
        // make sure this is not vacant;
        int vacant = 1;
        for (int n = 0; n < 20; n++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + n] != 0){
            vacant = 0;
            // printf("没有空啊\n");
            break;
          }
        }
        if (vacant) break;
        int time = (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 24] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 25];
        if ((time < prev_max) && (time > now_max)){
          now_max = time;
          now_max_index = j;
        }
      }

      char name[20];
      for (int j = 0; j < 20; j++){
        name[j] = fs->volume[fs->SUPERBLOCK_SIZE + now_max_index*(fs->FCB_SIZE) + j];
        if (name[j] == '\0') break;
      }
      if (fs->volume[fs->SUPERBLOCK_SIZE + now_max_index*(fs->FCB_SIZE) + 29] == 0){
        printf("[fs_gsys] : %s\n", name);
      }
      else if (fs->volume[fs->SUPERBLOCK_SIZE + now_max_index*(fs->FCB_SIZE) + 29] == 1){
        printf("[fs_gsys] : %s d\n", name);
      }
      else {
        printf("奇怪的bug\n");
      }
      
      prev_max = now_max;
    }
  }
  else if (op == LS_S){
    // list by size, same size -> first create first print
    int prev_max_size;
    int prev_min_time;
    int now_max_size;
    int now_min_time;
    int now_max_index;

    printf("[fs_gsys] : ===sort by file size===\n");
    for (int i = 0; i < fs->FCB_ENTRIES; i++) {
      int vacant = 1;
      for (int n = 0; n < 20; n++){
        if (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + n] != 0){
          vacant = 0;
          // printf("没有空啊\n");
          break;
        }
      }
      if (vacant) return;

      int complete = 1;
      if (i == 0) {
        prev_max_size = 0xffff;
        prev_min_time = -1;
      }
      for (int j = 0; j < fs->FCB_ENTRIES; j++) {
        int vacant = 1;
        for (int n = 0; n < 20; n++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + n] != 0){
            vacant = 0;
            // printf("没有空啊\n");
            break;
          }
        }
        if (vacant) break;

        int size = (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 23];
        int create_time = (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 26] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 27];
        if ((size < prev_max_size) || ((size == prev_max_size) && (create_time > prev_min_time))) {
          now_max_size = size;
          now_max_index = j;
          now_min_time = create_time;
          complete = 0;
          // printf("[fs_gsys LS] : found at least one candidate at %dth FCB. now_min_size = %d, now_min_time = %d\n", j, now_min_size, now_min_time);
          break;
        }
      }

      if (complete) {
        // printf("[fs_gsys LS] : LS_S complete.\n");
        return;
      }

      for (int j = 0; j < fs->FCB_ENTRIES; j++){
        int vacant = 1;
        for (int n = 0; n < 20; n++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + n] != 0){
            vacant = 0;
            // printf("没有空啊\n");
            break;
          }
        }
        if (vacant) break;
        int size = (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 23];
        int create_time = (fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 26] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + j*(fs->FCB_SIZE) + 27];
        if ((size < prev_max_size) || ((size == prev_max_size) && (create_time > prev_min_time))) {
          if ((size > now_max_size) || ((size == now_max_size) && (create_time < now_min_time))){
            now_max_size = size;
            now_max_index = j;
            now_min_time = create_time;
          }
        }
      }

      prev_max_size = now_max_size;
      prev_min_time = now_min_time;
      char name[20];
      for (int j = 0; j < 20; j++){
        name[j] = fs->volume[fs->SUPERBLOCK_SIZE + now_max_index*(fs->FCB_SIZE) + j];
        if (name[j] == '\0') break;
      }
      if (fs->volume[fs->SUPERBLOCK_SIZE + now_max_index*(fs->FCB_SIZE) + 29] == 0){
        printf("[fs_gsys] : %s %d\n", name, now_max_size);
      }
      else if (fs->volume[fs->SUPERBLOCK_SIZE + now_max_index*(fs->FCB_SIZE) + 29] == 1){
        printf("[fs_gsys] : %s %d d\n", name, now_max_size);
      }
      else {
        printf("奇怪的bug\n");
      }
      
    }
  }
  else if (op == CD_P){
    if ((fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE)] == '/') && (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 1] == '\0')){
      printf("[fs_gsys] : CD_P ERROR, already in the root dir.\n");
      return;
    }
    int parent_FCB_index = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 31];
    current_dir_index = parent_FCB_index;
  }
  else if (op == PWD){
    char pwd_name[20];
    int current_depth = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 28];

    for (int i = 0; i < 20; i++){
      pwd_name[i] = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + i];
      if (pwd_name == '\0') break;
    }
    if (current_depth == 1){
      printf("[fs_gsys] : %s\n", pwd_name);
    }
    else if (current_depth == 2) {
      printf("[fs_gsys] : /%s\n", pwd_name);
    }
    else if (current_depth == 3) {
      char pwd_upper_name[20];
      int parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 31];
      for (int i = 0; i < 20; i++){
        pwd_upper_name[i] = fs->volume[fs->SUPERBLOCK_SIZE + parent_index*(fs->FCB_SIZE) + i];
        if (pwd_upper_name[i] == '\0') break;
      }
      printf("[fs_gsys] : /%s/%s\n", pwd_upper_name, pwd_name);
    }
    
  }
  else printf("[fs_gsys ERROR] : unknown op.\n");
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  /* search file name in FCB */
  int found = 0;
  u32 fp;
  int FCB_index;
  int block_index;
  int length;
  int block_size;

  if (op == RM){
  // printf("REMOVE前, VCB[4092]: %x\n", fs->volume[4092]);

    for (int i = 0; i < fs->FCB_ENTRIES; i++){
      if (!found){
        if (((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31]) != current_dir_index) continue;
        for (int j = 0; j < fs->FCB_SIZE; j++){
          if (s[j] != fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) break;
          else if ((s[j] == fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) && (s[j] == '\0') ) {
            // if found the file, return
            found = 1;
            FCB_index = i;
            block_index = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 21];
            length = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 23];
            block_size = (length == 0) ? 1 : (1 + (length-1) / fs->STORAGE_BLOCK_SIZE);
            fp = fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 20]) << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 21]);
            printf("[fs_gsys REMOVE] : Remove existing file: %s, fp is %x, FCB = %d, start block index = %x, length = %d, time = %d\n", 
              s, fp, i, block_index, length,
              (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 24] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 25]);
            break;
          }
        }
      }
      else break;
    }
    if (!found){
      printf("[fs_gsys ERROR] : RM op. no file to remove.\n");
      return;
    }
    
    /* do the compaction */

    // compact the contents of file
    for (int i = 0; i < fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE; i++){
      if (block_index + i == (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE)) break;
      int VCB_index = (block_index + i) / 8;
      int VCB_offset = (block_index + i) % 8;
      if ((fs->volume[VCB_index] & (1 << (7 - VCB_offset))) != 0){

        for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
          if (block_index + i +block_size >= (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE)) {
            fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(block_index + i) + j] = 0;
          }
          else fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(block_index + i) + j] = fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(block_index + i +block_size) + j];
        }
      }
      else break;
    }

    // compact the VCB (from the backward direction)
    int last_block = (1 << 15);
    for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
      int VCB_index = i / 8;
      int VCB_offset = i % 8;
      if ((fs->volume[VCB_index] & (1 << (7-VCB_offset))) == 0) {
        last_block = i - 1;
        // printf("最后一个block index为 %d\n", last_block);
        break;
      }
    }
    for (int i = 0; i < block_size; i++){
      int VCB_index = (last_block - i) / 8;
      int VCB_offset = (last_block - i) % 8;
      fs->volume[VCB_index] &= (0xff - (1 << (7 - VCB_offset)));
      // printf("压VCB: VCB[%d]: %x\n", VCB_index, fs->volume[VCB_index]);
    }

    // compact the FCB
    for (int i = 0; i < fs->FCB_ENTRIES; i++){
      if (FCB_index + i + 1 == fs->FCB_ENTRIES) {
        for (int j = 0; j < fs->FCB_SIZE; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = 0;
        }
        break;
      }

      int vacant_next_FCB = 1;
      for (int j = 0; j < 20; j++){
        if (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + j] != 0) {
          vacant_next_FCB = 0;
          break;
        }
      }
      if (vacant_next_FCB){
        for (int j = 0; j < fs->FCB_SIZE; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = 0;
        }
        break;
      }
      int prev_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 21];
      int new_start_block = prev_start_block - block_size;
      printf("FCB index = %d, i = %d, prev_start_block = %x, new_start_block = %x \n", FCB_index, i, prev_start_block, new_start_block);
      // update name
      char new_name[20];
      for (int j = 0; j < 20; j++){
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i + 1)*(fs->FCB_SIZE) + j];
        new_name[j] = fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j];
      }

      // update start block index
      fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 20] = (new_start_block >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 21] = (new_start_block & 0xff);

      // update length, modification time, create time, depth, identity.
      for (int j = 22; j < 30; j++){
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + j];
      }

      // update second half parents
      int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + 31];
      if (this_parent_index == 0xffff){
        fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
        fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
      }
      else if (this_parent_index > FCB_index){
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
      }
      else{
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 30] = (this_parent_index >> 8);
        fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 31] = (this_parent_index & 0xff);
      }

      
      printf("Update file: name = %s, FCB = %d, start block index = %x, length = %d, time = %d\n", 
        new_name, FCB_index + i,
        (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 21],
        (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 23],
        (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 24] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + 25]);
    }

    // update first half parents
    for (int i = 0; i < FCB_index; i++){
      int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31];
      if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
      }
      else if (this_parent_index > FCB_index){
        fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
        fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
      }
    }

    // update the current dir.
    if (current_dir_index > FCB_index) current_dir_index--;

    /* update the parent directory information*/
    int old_dir_length = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23];
    int old_dir_block_size = (old_dir_length == 0) ? 1 : 1 + (old_dir_length - 1)/fs->STORAGE_BLOCK_SIZE;
    int delete_file_size = 0;
    for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++){
      if (s[i] == '\0'){
        delete_file_size += i+1;
        break;
      }
    }
    int new_dir_length = old_dir_length - delete_file_size;
    int new_dir_block_size = (new_dir_length == 0) ? 1 : 1 + (new_dir_length - 1)/fs->STORAGE_BLOCK_SIZE;
    int old_dir_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 21];


    if (new_dir_block_size == old_dir_block_size){
      /* if size does not change, no need to compact. */
      
      // update the dir FCB
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] = (new_dir_length >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23] = (new_dir_length & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

      // update the dir file contents
      int found = 0;
      for (int i = 0; i < old_dir_block_size*(fs->STORAGE_BLOCK_SIZE); i++){
        if (!found){
          for (int j = 0; j < delete_file_size; j++){
            if (fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i + j] != s[j]) break;
            if (j == delete_file_size - 1) found = 1;
          }
        }
        if (found){
          if (i < new_dir_length) fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i] = fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i + delete_file_size];
          else fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i] = 0;
        }
      }
    }

    else{
      /* if different block size, push all the later files forward. */

      // write the contents first
      int found = 0;
      for (int i = 0; i < new_dir_block_size*(fs->STORAGE_BLOCK_SIZE); i++){
        if (!found){
          for (int j = 0; j < delete_file_size; j++){
            if (fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i + j] != s[j]) break;
            if (j == delete_file_size - 1) found = 1;
          }
        }
        if (found){
          if (i < new_dir_length) fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i] = fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i + delete_file_size];
          else fs->volume[fs->FILE_BASE_ADDRESS + old_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i] = 0;
        }
      }

      // update the dir FCB
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] = (new_dir_length >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23] = (new_dir_length & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);      

      /* compact the later files forward. */
      // compact the file contents
      int start_compact_block = old_dir_start_block + new_dir_block_size;
      int block_diff = old_dir_block_size - new_dir_block_size;
      printf("block_diff is %d\n", block_diff);
      for (int i = start_compact_block; i < (1 << 15); i++) {
        int VCB_index = (i) / 8;
        int VCB_offset = (i) % 8;
        if ((fs->volume[VCB_index] & (1 << (7 - VCB_offset))) != 0){
          if (i + block_diff >= (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE)){
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(i) + j] = 0;
            }
          }
          else {
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(i) + j] = fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(i + block_diff) + j];
            }
          }
        }
        else break;
      }

      // update the VCB (from backward)
      int last_block = (1 << 15);
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = i / 8;
        int VCB_offset = i % 8;
        if ((fs->volume[VCB_index] & (1 << (7-VCB_offset))) == 0) {
          last_block = i - 1;
          // printf("最后一个block index为 %d\n", last_block);
          break;
        }
      }
      for (int i = 0; i < block_diff; i++){
        int VCB_index = (last_block - i) / 8;
        int VCB_offset = (last_block - i) % 8;
        fs->volume[VCB_index] &= (0xff - (1 << (7 - VCB_offset)));
        // printf("压VCB: VCB[%d]: %x\n", VCB_index, fs->volume[VCB_index]);
      }

      // compact the FCB
      for (int i = 0; i < fs->FCB_ENTRIES; i++){
        int vacant_this_FCB = 1;
        for (int j = 0; j < 20; j++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + j] != 0) {
            vacant_this_FCB = 0;
            break;
          }
        }
        if (vacant_this_FCB){
          break;
        }

        if (i > current_dir_index){
          int prev_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + 21];
          int new_start_block = prev_start_block - block_diff;
          fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + 20] = (new_start_block >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + 21] = (new_start_block & 0xff);
        }

        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index > current_dir_index) {
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = ((this_parent_index - 1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = ((this_parent_index - 1) & 0xff);
        }
      }
      /* compaction finished */
    }
    
  }
  else if (op == MKDIR){
      /* Searching the dir name in FCB. Make sure no reprtitive subdir in the same dir. */
    int current_depth = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 28];
    if (current_depth >= 3) {
      printf("[ERROR MKDIR] : depth is over the limit.\n");
      return;
    }
    int first_avai_FCB = 1023;
    for (int i = 0; i < fs->FCB_ENTRIES; i++){
        int vacant_this_FCB = 1;
        for (int j = 0; j < 20; j++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + j] != 0) {
            vacant_this_FCB = 0;
            break;
          }
        }
        if (vacant_this_FCB){
          first_avai_FCB = i;
          break;
        }
        if (((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31]) != current_dir_index) continue;
        for (int j = 0; j < fs->FCB_SIZE; j++){
          if (s[j] != fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) break;
          else if ((s[j] == fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) && (s[j] == '\0')) {
            printf("[fs_gsys ERROR MKDIR] : already exist subdirectory in this directory.\n");
            return;
          }
        }
    }
    /* traverse the VCB to record the first available block */
    int first_avai_block = 0xffff;
    for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++){
      if (fs->volume[i] != 0xff){
        for (int j = 7; j >= 0; j--){
          if (((fs->volume[i] & (1 << j)) >> j) == 0) {
            // record this block
            first_avai_block = i*8 + (7-j);
            fs->volume[i] += (1 << j);
            break;
          }
        }
        break;
      }
    }
    if (first_avai_block == 0xffff) {
      printf("[ERROR fs_gsys MKDIR] : no available block.\n");
      return;
    }
    
    /* write the FCB */
    for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++){
      fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + i] = s[i];
      if (s[i] == '\0') break;
    }

    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 20] = (first_avai_block >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 21] = (first_avai_block & 0xff);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 24] = (gtime >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 25] = (gtime & 0xff);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 26] = (gtime >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 27] = (gtime & 0xff);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 28] = current_depth + 1;
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 29] = 1;
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 30] = (current_dir_index >> 8);
    fs->volume[fs->SUPERBLOCK_SIZE + first_avai_FCB*(fs->FCB_SIZE) + 31] = (current_dir_index & 0xff);

    /* write the current dir FCB modification time*/
    /* write the parent dir */
    int name_length = 0;
    for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
      if (s[i] == '\0') {
        name_length = i+1;
        break;
      }
    }
    int parent_dir_length = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23];
    int parent_dir_block_size = (parent_dir_length == 0) ? 1 : 1 + (parent_dir_length - 1) / fs->STORAGE_BLOCK_SIZE;
    int parent_dir_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 21];
    int new_parent_dir_length = parent_dir_length + name_length;
    int new_parent_dir_block_size = (new_parent_dir_length == 0) ? 1 : 1 + (new_parent_dir_length - 1) / fs->STORAGE_BLOCK_SIZE;
    int parent_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 31];
    char new_dir_contents[1024];
    char parent_dir_name[20];
    char parent_dir_invariant[4];

    if (parent_dir_block_size == new_parent_dir_block_size) {
      /* update FCB of parent dir */
        // length
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 22] = (new_parent_dir_length >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 23] = (new_parent_dir_length & 0xff);
        // modification time
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 25] = (gtime & 0xff);

      /* update the contents of parent dir */
      for (int i = 0; i < name_length; i++){
        fs->volume[fs->FILE_BASE_ADDRESS + parent_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + parent_dir_length + i] = s[i];
      }
    }
    else{
      for (int i = 0; i < parent_dir_length; i++){
        new_dir_contents[i] = fs->volume[fs->FILE_BASE_ADDRESS + parent_dir_start_block*(fs->STORAGE_BLOCK_SIZE) + i];
      }
      for (int i = 0; i < name_length; i++){
        new_dir_contents[parent_dir_length + i] = s[i];
      }
      for (int i = 0; i < 20; i++){
        parent_dir_name[i] = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + i];
        if (parent_dir_name[i] == '\0') break;
      }
      for (int i = 0; i < 4; i++){
        parent_dir_invariant[i] = fs->volume[fs->SUPERBLOCK_SIZE + current_dir_index*(fs->FCB_SIZE) + 26 + i];
      }

      /* do the compaction */
      int new_avai_FCB = fs->MAX_FILE_NUM - 1;
      int new_avai_block = (1 << 15) - 1;

      // compact the contents of file
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = (parent_dir_start_block + i) / 8;
        int VCB_offset = (parent_dir_start_block + i) % 8;
        if ((fs->volume[VCB_index] & (1 << (7 - VCB_offset))) != 0){
          if (parent_dir_start_block + i + parent_dir_block_size >= (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE)){
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(parent_dir_start_block + i) + j] = 0;
            }
          }
          else {
            for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
              fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(parent_dir_start_block + i) + j] = fs->volume[fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE*(parent_dir_start_block + i +parent_dir_block_size) + j];
            }
          }
        }
        else {
          new_avai_block = parent_dir_start_block + i - parent_dir_block_size;
          break;
        }
      }
      
      // compact the VCB (from the backward direction)
      // find the last allocated block
      int last_block = (1 << 15);
      for (int i = 0; i < (fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE); i++){
        int VCB_index = i / 8;
        int VCB_offset = i % 8;
        if ((fs->volume[VCB_index] & (1 << (7-VCB_offset))) == 0) {
          last_block = i - 1;
          // printf("write compact: 最后一个block index为 %d\n", last_block);
          break;
        }
      }
      
      for (int i = 0; i < parent_dir_block_size; i++){
        int VCB_index = (last_block - i) / 8;
        int VCB_offset = (last_block - i) % 8;
        fs->volume[VCB_index] &= (0xff - (1 << (7 - VCB_offset)));
        // printf("压VCB: VCB[0], VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      }
      

      // compact the FCB
      for (int i = 0; i < fs->FCB_ENTRIES; i++){
        if (current_dir_index + i + 1 >= fs->FCB_ENTRIES) {
          for (int j = 0; j < fs->FCB_SIZE; j++){
            fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = 0;
          }
          break;
        }

        int vacant_next_FCB = 1;
        for (int j = 0; j < 20; j++){
          if (fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i+1)*(fs->FCB_SIZE) + j] != 0) {
            vacant_next_FCB = 0;
            break;
          }
        }
        if (vacant_next_FCB){
          for (int j = 0; j < fs->FCB_SIZE; j++){
            fs->volume[fs->SUPERBLOCK_SIZE + (FCB_index + i)*(fs->FCB_SIZE) + j] = 0;
          }
          new_avai_FCB = i;
          break;
        }

        int prev_start_block = (fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 20] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 21];
        int new_start_block = prev_start_block - parent_dir_block_size;
        
        // update name
        for (int j = 0; j < 20; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i + 1)*(fs->FCB_SIZE) + j];
        }

        // update start block index
        fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 20] = (new_start_block >> 8);
        fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 21] = (new_start_block & 0xff);
        
        // update length, modification time, create time, depth, identity
        for (int j = 22; j < 30; j++){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + j] = fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + j];
        }

        // update second half parents
        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i+1)*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index == current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (new_avai_FCB >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (new_avai_FCB & 0xff);
        }
        else if (this_parent_index > current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
        }
        else{
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (this_parent_index >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (this_parent_index & 0xff);
        }
      }
      // update first half parents
      for (int i = 0; i < current_dir_index; i++){
        int this_parent_index = (fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31];
        if (this_parent_index == 0xffff){
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 30] = (0xff);
          fs->volume[fs->SUPERBLOCK_SIZE + (current_dir_index + i)*(fs->FCB_SIZE) + 31] = (0xff); 
        }
        else if (this_parent_index == current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = (new_avai_FCB >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = (new_avai_FCB & 0xff);
        }
        else if (this_parent_index > current_dir_index){
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] = ((this_parent_index-1) >> 8);
          fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31] = ((this_parent_index-1) & 0xff);
        }
      }
      /* compaction finished */
      /* write new parent dir */

        // write to VCB
      int new_VCB_index;
      int new_VCB_offset;
      for (int i = 0; i < new_parent_dir_block_size; i++){
        new_VCB_index = (i + new_avai_block) / 8;
        new_VCB_offset = (i + new_avai_block) % 8;
        fs->volume[new_VCB_index] |= (1 << (7 - new_VCB_offset));
        // printf("写VCB: VCB[0], VCB[1]: %x %x\n", fs->volume[0], fs->volume[1]);
      }
      
      
      // write to FCB
      for (int i = 0; i < 20; i++) {
        fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + i] = parent_dir_name[i];
        if (parent_dir_name[i] == '\0') {
          break;
        }
      }
      
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 20] = (new_avai_block >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 21] = (new_avai_block & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 22] = (new_parent_dir_length >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 23] = (new_parent_dir_length & 0xff);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 24] = (gtime >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 25] = (gtime & 0xff);
      for (int i = 0; i < 4; i++){
        fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 26 + i] = parent_dir_invariant[i];
      }
        // update parent in FCB
      if (parent_parent_index > current_dir_index) {
        if (parent_parent_index != 0xffff){
          parent_parent_index--;
        }
        
      }
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 30] = (parent_parent_index >> 8);
      fs->volume[fs->SUPERBLOCK_SIZE + new_avai_FCB*(fs->FCB_SIZE) + 31] = (parent_parent_index & 0xff);
      
      // write to file content
      for (int i = 0; i < new_parent_dir_length; i++){
        fs->volume[fs->FILE_BASE_ADDRESS + new_avai_block*(fs->STORAGE_BLOCK_SIZE) + i] = new_dir_contents[i];
      }
      
      // update current dir index
      current_dir_index = new_avai_FCB;
    }
  }

  else if (op == CD) {

    for (int i = 0; i < fs->FCB_ENTRIES; i++){
      int vacant_this_FCB = 1;
      for (int j = 0; j < 20; j++){
        if (fs->volume[fs->SUPERBLOCK_SIZE + (i)*(fs->FCB_SIZE) + j] != 0) {
          vacant_this_FCB = 0;
          break;
        }
      }
      if (vacant_this_FCB){
        break;
      }
      if (((fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 30] << 8) + fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + 31]) != current_dir_index) continue;
      for (int j = 0; j < fs->FCB_SIZE; j++){
        if (s[j] != fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) break;
        else if ((s[j] == fs->volume[fs->SUPERBLOCK_SIZE + i*(fs->FCB_SIZE) + j]) && (s[j] == '\0')) {
          printf("[fs_gsys CD] : find it.\n");
          current_dir_index = i;
          return;
        }
      }
    }
  }

  else if (op == RM_RF) {
    printf("[SORRY] : no time to implement it. I have drafted my implementation plan in the report\n");
    return;
  }
  else {
    printf("[fs_gsys ERROR] : no such op.\n");
  }
  gtime++;
}