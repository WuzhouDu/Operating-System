#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

typedef struct thread_pool_task{
  void (*task_function)(int);
  int args;
  struct thread_pool_task* next;
} pool_task_t;

typedef struct thread_pool{
  int  max_thread;
  pthread_t* thread_ids;
  pool_task_t*  thread_pool_head;
  pthread_mutex_t queue_lock;
  pthread_cond_t  queue_ready;
} thread_pool_t;

void async_init(int);
void async_run(void (*fx)(int), int args);

#endif
