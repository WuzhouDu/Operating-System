#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <stdint.h>

#define ROW 10
#define COLUMN 50 
#define LOGLEN 15


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ; 

// pthread_mutex_t status_lock;
pthread_mutex_t frog_lock;
pthread_mutex_t map_lock;
int status = 0;


// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){
	int id = (intptr_t) t;
	int begin_index;
	int end_index;
	// printf("I'm thread %d\n", id); /* there are totally 11 new threads, from id = 0 to id = 10 */
	/*  Move the logs  */
	if (id != 0 && id != 10){
		int initial_place = rand() % 49;
		for (int i = 0; i < LOGLEN; ++i){
			pthread_mutex_lock(&map_lock);
			map[id][(initial_place+i)%49] = '=';
			pthread_mutex_unlock(&map_lock);
		}
		begin_index = initial_place;
		end_index = (begin_index + 14)%49;
	}
	
	printf("\033[?25l");
	while(!status){
		if (id == 10){
			printf("\033[%d;1H%s", id+1, map[10]);
			usleep(50000);
			continue;
		}

		else if (id == 0){
			printf("\033[%d;1H%s", id+1, map[0]);
			if (frog.x == id){
				status = 1;
				continue;
			}
			usleep(50000);
			continue;
		}

		else if (!(id%2)){ // in 2, 4, 6, 8 line, log moves right
			printf("\033[%d;1H%s", id+1, map[id]);
			char temp[COLUMN-1];
			for (int i = 0; i < COLUMN-1; ++i){
				temp[i] = map[id][(i-1+49)%49];
			}

			begin_index = (begin_index+1)%49;
			end_index = (begin_index + 14)%49;
			
			pthread_mutex_lock(&map_lock);
			strcpy(map[id], temp);
			pthread_mutex_unlock(&map_lock);

			if (frog.x == id){
				pthread_mutex_lock(&frog_lock);
				frog.y++;
				pthread_mutex_unlock(&frog_lock);

				if ((begin_index < end_index && (frog.y < begin_index || frog.y > end_index)) || 
					(begin_index > end_index && (frog.y > end_index && frog.y < begin_index)) ||
					(frog.y == 49)){
					status = 2;
					continue;
				}
			}
			usleep(50000);
		}
		
		else if (id%2){ // in 1, 3, 5, 7, 9 line, log moves left
			printf("\033[%d;1H%s", id+1, map[id]);
			char temp[COLUMN-1];
			for (int i = 0; i < COLUMN-1; ++i){
				temp[i] = map[id][(i+1+49)%49];
			}

			begin_index = (begin_index-1+49)%49;
			end_index = (begin_index+14)%49;
			
			pthread_mutex_lock(&map_lock);
			strcpy(map[id], temp);
			pthread_mutex_unlock(&map_lock);

			if (frog.x == id){
				pthread_mutex_lock(&frog_lock);
				frog.y--;
				pthread_mutex_unlock(&frog_lock);

				if ((begin_index < end_index && (frog.y < begin_index || frog.y > end_index)) || 
					(begin_index > end_index && (frog.y > end_index && frog.y < begin_index)) ||
					(frog.y == -1)){
					// pthread_mutex_lock(&status_lock);
					status = 2;
					// pthread_mutex_unlock(&status_lock);
					continue;
				}
			}
			usleep(50000);
		}
	}
	printf("\033[2J");
	pthread_exit(NULL);	
}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	printf("\033[2J\033[H\033[?25l");
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );
	
	if (pthread_mutex_init(&frog_lock, NULL)){
		printf("frog_lock init fails\n");
		exit(1);
	}

	if (pthread_mutex_init(&map_lock, NULL)){
		printf("map_lock init fails\n");
		exit(1);
	}	

	pthread_t threads[ROW+1];
	int thread_create_status;
	for ( i = 0; i < ROW+1; ++i){
		thread_create_status = pthread_create(&threads[i], NULL, logs_move, (void*) (intptr_t)i);
		if(thread_create_status){
			printf("ERROR: return status from pthread_create: %d", thread_create_status);
			exit(1);
		}
	}

	
	/*  Display the output for user: win, lose or quit.  */


	while(!status){
		if (kbhit()){
			char input = getchar();
			if (input == 'q' || input == 'Q'){
				// pthread_mutex_lock(&status_lock);
				status = 3;
				// pthread_mutex_unlock(&status_lock);
			}
			else if (input == 'w' || input == 'W'){
				pthread_mutex_lock(&frog_lock);
				--frog.x;
				pthread_mutex_unlock(&frog_lock);

				pthread_mutex_lock(&map_lock);
				map[frog.x][frog.y] = '0';
				map[frog.x+1][frog.y] = (frog.x+1 == 10) ? '|' : '=';
				pthread_mutex_unlock(&map_lock);

				// printf("after type in w, frog:%d, %d\n", frog.x, frog.y);
			}
			else if (input == 's' || input == 'S'){
				if (frog.x == 10){
					// pthread_mutex_lock(&status_lock);
					status = 2;
					// pthread_mutex_unlock(&status_lock);
					break;
				}
				pthread_mutex_lock(&frog_lock);
				++frog.x;
				pthread_mutex_unlock(&frog_lock);

				pthread_mutex_lock(&map_lock);
				map[frog.x][frog.y] = '0';
				map[frog.x-1][frog.y] = '=';
				pthread_mutex_unlock(&map_lock);

				// printf("after type in w, frog:%d, %d\n", frog.x, frog.y);
			}
			else if (input == 'a' || input == 'A'){
				if (frog.y == 0){
					// pthread_mutex_lock(&status_lock);
					status = 2;
					// pthread_mutex_unlock(&status_lock);
					break;
				}
				pthread_mutex_lock(&frog_lock);
				--frog.y;
				pthread_mutex_unlock(&frog_lock);

				pthread_mutex_lock(&map_lock);
				map[frog.x][frog.y] = '0';
				map[frog.x][frog.y+1] = (frog.x == 10) ? '|' : '=';
				pthread_mutex_unlock(&map_lock);

				// printf("after type in a, frog:%d, %d\n", frog.x, frog.y);
			}
			else if (input == 'd' || input == 'D'){
				if (frog.y == COLUMN-2){
					// pthread_mutex_lock(&status_lock);
					status = 2;
					// pthread_mutex_unlock(&status_lock);
					break;
				}				
				pthread_mutex_lock(&frog_lock);
				++frog.y;
				pthread_mutex_unlock(&frog_lock);

				pthread_mutex_lock(&map_lock);
				map[frog.x][frog.y] = '0';
				map[frog.x][frog.y-1] = (frog.x == 10) ? '|' : '=';
				pthread_mutex_unlock(&map_lock);

				// printf("after type in d, frog:%d, %d\n", frog.x, frog.y);
			}
		}
	}

	for (i = 0; i < ROW+1; ++i){
		pthread_join(threads[i], NULL);
	}
	switch (status){
		case 1: printf("\033[2J\033[HYou win!!!!\n"); break;
		case 2: printf("\033[2J\033[HYou lose!!!\n"); break;
		case 3: printf("\033[2J\033[HYou quit the game.\n"); break;
	}


	// pthread_mutex_destroy(&status_lock);
	pthread_mutex_destroy(&frog_lock);
	pthread_exit(NULL);
	return 0;

}