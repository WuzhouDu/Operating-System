#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include "async.h"
#include "utlist.h"

static thread_pool_t pool;

void* routine_fun(void* args){
    pool_task_t* task;
    while(1){
        pthread_mutex_lock(&(pool.queue_lock));
        while (!pool.thread_pool_head){
            pthread_cond_wait(&(pool.queue_ready), &(pool.queue_lock));
        }
        task = pool.thread_pool_head;
        pool.thread_pool_head = pool.thread_pool_head->next;
        pthread_mutex_unlock(&(pool.queue_lock));
        task->task_function(task->args);
        free(task);
    }

    return NULL;
}


void async_init(int num_threads) {    
    pool.max_thread = num_threads;
    pool.thread_ids = (pthread_t*)malloc(sizeof(pthread_t)*num_threads);
    pool.thread_pool_head = NULL;

    if (pool.thread_ids == NULL){
        printf("ERROR: malloc thread idsfails.\n");
        exit(1);
    }
    if (pthread_mutex_init(&(pool.queue_lock), NULL) != 0){
        printf("ERROR: queue_lock init fails.\n");
        exit(1);
    }
    if (pthread_cond_init(&(pool.queue_ready), NULL) != 0){
        printf("ERROR: queue_ready init fails.\n");
        exit(1);
    }
    for (int i = 0; i < num_threads; ++i){
        if (pthread_create(&(pool.thread_ids[i]), NULL, routine_fun, NULL) != 0){
            printf("pthread_create fails!\n");
            exit(1);
        }
    }

    return;
}

void async_run(void (*handler)(int), int args) {
    pool_task_t* task = malloc(sizeof(pool_task_t));
    pool_task_t* queue_end;

    if (!handler){
        printf("ERROR: no handler function in task!\n");
        exit(1);
    }

    task->task_function = handler;
    task->args = args;
    task->next = NULL;

    pthread_mutex_lock(&(pool.queue_lock));
    queue_end = pool.thread_pool_head;
    if (queue_end == NULL){
        pool.thread_pool_head = task;
    }
    else{
        while(queue_end->next){
            queue_end = queue_end->next;
        }
        queue_end->next = task;
    }
    pthread_cond_signal(&(pool.queue_ready));
    pthread_mutex_unlock(&(pool.queue_lock));
    // handler(args);
    return;
}