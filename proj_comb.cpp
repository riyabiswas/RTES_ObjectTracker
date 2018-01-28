#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <syslog.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <semaphore.h>


using namespace cv;
using namespace std;

#define HRES 640
#define VRES 480
#define OK 1
#define NSEC_PER_SEC 1000000000
//global variables
pthread_t  Thread_frame, Thread_center, Thread_pred;
pthread_attr_t attr_frame, attr_center, attr_pred, main_attr; // make one attr for main
struct sched_param main_param, rt1_param, rt2_param, rt3_param; // make one param for main
CvCapture* capture;
IplImage* frame_new;
IplImage* frame_old;

int dev=0;
pthread_mutex_t lock;
struct timespec time1, time2, time4, time5, time6, time7, old_time_cent, new_time_cent, dif_cent, difference;
char frame_captured[] = "Circle detection";

int cent_x_new;
int cent_y_new;
int cent_x_old;
int cent_y_old;

double speed_x;
double speed_y;

double dist_x;
double dist_y;

double nextPos_x;
double nextPos_y;

/*Define semaphores*/
sem_t sem_frame, sem_center, sem_pred;

int center_count;
int frame_counter;


/////////////////////////////////////////////////////////////////

// function to calculate the differnece in time

////////////////////////////////////////////////////////////////

int time_difference(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }

  return(OK);
}

/////////////////////////////////////////////////////////////////

//Function to calculate centroid of the circle

////////////////////////////////////////////////////////////////

void * center(void* b)
{
	
    int x_new=0;
    int y_new=0;
	
    int x_old=0;
    int y_old=0;
    int counter=0;
   
    Mat gray_new, gray_old;
    vector<Vec3f> circles_new, circles_old;
    while(1)
    {   
		/*Hold semaphore*/
	    sem_wait(&sem_center);
clock_gettime(CLOCK_REALTIME, &time1);
syslog(LOG_INFO, "Center cal starts at sec= %ld  nsec= %ld \n", time1.tv_sec, time1.tv_nsec);

	pthread_mutex_lock(&lock);
    //	clock_gettime(CLOCK_REALTIME, &time5);
    	Mat mat_frame_new(frame_new);
    	Mat mat_frame_old(frame_old);
    	pthread_mutex_unlock(&lock);

        cvtColor(mat_frame_new, gray_new, CV_BGR2GRAY);
        cvtColor(mat_frame_old, gray_old, CV_BGR2GRAY);

        GaussianBlur(gray_new, gray_new, Size(9,9), 2, 2);
	GaussianBlur(gray_old, gray_old, Size(9,9), 2, 2);

        HoughCircles(gray_new, circles_new, CV_HOUGH_GRADIENT, 1, gray_new.rows/8, 100, 50, 0, 0);
        HoughCircles(gray_old, circles_old, CV_HOUGH_GRADIENT, 1, gray_old.rows/8, 100, 50, 0, 0);
	
	syslog(LOG_DEBUG, "circles.size = %d\n", circles_new.size());
	syslog(LOG_DEBUG, "circles.size = %d\n", circles_old.size());
	
		for( size_t i = 0; i < circles_new.size(); i++ )
		{
			
		  Point center(cvRound(circles_new[i][0]), cvRound(circles_new[i][1]));
		  int radius = cvRound(circles_new[i][2]);
		  x_new= cvRound(circles_new[i][0]);
	   	  y_new= cvRound(circles_new[i][1]);	

	   	 // printing the center cordinates
	   	  syslog(LOG_DEBUG,"Current centroid_new position X: %d, Y:%d\n", x_new,y_new);
	   	  //syslog(LOG_DEBUG,"Y coordinate of the current circle: %d\n", y);
		  // circle center
		  circle( mat_frame_new, center, 3, Scalar(0,255,0), -1, 8, 0 );
		  // circle outline
		  circle( mat_frame_new, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}
		for( size_t i = 0; i < circles_old.size(); i++ )
		{
			
		  Point center(cvRound(circles_old[i][0]), cvRound(circles_old[i][1]));
		  int radius = cvRound(circles_old[i][2]);
		  x_old= cvRound(circles_old[i][0]);
	   	  y_old= cvRound(circles_old[i][1]);	

	   	 // printing the center cordinates
	   	  syslog(LOG_DEBUG,"Current centroid_old position X: %d, Y:%d\n", x_old,y_old);
	   	  //syslog(LOG_DEBUG,"Y coordinate of the current circle: %d\n", y);
		  // circle center
		  circle( mat_frame_old, center, 3, Scalar(0,255,0), -1, 8, 0 );
		  // circle outline
		  circle( mat_frame_old, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}	

		// Saving the previous time stamp
			old_time_cent =new_time_cent;
			// Fetching the new time
			clock_gettime(CLOCK_REALTIME, &new_time_cent);
			syslog(LOG_DEBUG,"Time at which this centroid calculated =%ld second and %ld nanosecond\n", new_time_cent.tv_sec, new_time_cent.tv_nsec);
			// Saving the previoud cordinates of the center
			cent_x_old = cent_x_new;
			cent_y_old = cent_y_new;
			// Fetching the new cordinates of the center	
			cent_x_new = (x_old+x_new)/2;
			cent_y_new = (y_old+y_new)/2;

			imshow(frame_captured, mat_frame_new);
			imshow(frame_captured, mat_frame_old);
	       		 char c = cvWaitKey(10);  			
		center_count = (center_count+1)%2; 
	    	if(center_count == 0)
	    	{   
		   	sem_post(&sem_pred);	
    		}	
		else
		  sem_post(&sem_frame);


	clock_gettime(CLOCK_REALTIME, &time2);
syslog(LOG_INFO, "Center cal ends at sec= %ld  nsec= %ld \n", time2.tv_sec, time2.tv_nsec);
time_difference(&time2, &time1, &difference);
	syslog(LOG_INFO, "Diff in time = %ld  nsec= %ld \n", difference.tv_sec, difference.tv_nsec);
	
  }// while
}

/////////////////////////////////////////////////////////////////

//Function to capture the frames

////////////////////////////////////////////////////////////////

void * frame_capture(void * a)
{
	
	
	
  while(1){
   		/*Hold semaphore for frame capture*/
		sem_wait(&sem_frame);
clock_gettime(CLOCK_REALTIME, &time4);
syslog(LOG_INFO, "frame cap start at sec= %ld  nsec= %ld \n", time4.tv_sec, time4.tv_nsec);
		/*Critical section begins*/
	    pthread_mutex_lock(&lock);
	    frame_old = frame_new;
	     frame_new=cvQueryFrame(capture);
	     if(!frame_new) break;  
	    pthread_mutex_unlock(&lock);
		/*Critical section ends*/
	    frame_counter++;
	   syslog(LOG_DEBUG,"Number of frame = %d\n", frame_counter);

        if(frame_counter == 4)
   	 {   
		//Resetting the frame count
		frame_counter = 0;
		/*Release semaphore for centroid calculation*/
		sem_post(&sem_center);
	}
	else
	{
		/*Release semaphore for frame capture*/
		sem_post(&sem_frame);
	}
	clock_gettime(CLOCK_REALTIME, &time5);
time_difference(&time5, &time4, &difference);
	
	syslog(LOG_INFO, "frame cap ends at sec= %ld  nsec= %ld \n", time5.tv_sec, time5.tv_nsec);
	syslog(LOG_INFO, "Difference = %ld  nsec= %ld \n", difference.tv_sec, difference.tv_nsec);
	
    }// while

} // frame capture funtion


/////////////////////////////////////////////////////////////////

//Fucntion predicting the next position of the centroid

////////////////////////////////////////////////////////////////

double time_dif = 0;

void * predict_position(void *threadid)
{Mat final_plot=Mat::zeros( 480, 640, CV_8UC3 );
	while(1)
 {
	
 	/*Hold semaphore*/
   	 sem_wait(&sem_pred);
clock_gettime(CLOCK_REALTIME, &time6);
syslog(LOG_INFO, "Pred starts at sec= %ld  nsec= %ld \n", time6.tv_sec, time6.tv_nsec);
	dist_x = abs(cent_x_new-cent_x_old);
	dist_y = abs(cent_y_new-cent_y_old);
	time_difference(&new_time_cent, &old_time_cent, &dif_cent);
	time_dif = (dif_cent.tv_sec*1000000) + dif_cent.tv_nsec/1000; // in miliseconds
	speed_x = dist_x/time_dif;
	speed_y = dist_y/time_dif;

	nextPos_x = cent_x_new + speed_x*((double)time_dif);
	nextPos_y = cent_y_new + speed_y*((double)time_dif);
	
//	final_plot.at<Vec3b>(nextPos_x,nextPos_y) = Vec3b(255, 150, 150);
	 Point center(cvRound(nextPos_x), cvRound(nextPos_y));
	circle( final_plot, center, 3, Scalar(0,255,0), -1, 8, 0 );
	imshow("Final_Plot", final_plot);
	char c = cvWaitKey(1);
	
 	syslog(LOG_INFO, "Predicted centroid position X:%f, Y:%f \n",nextPos_x,nextPos_y);
	/*Release semaphore*/
	clock_gettime(CLOCK_REALTIME, &time7);
syslog(LOG_INFO, "Pred ends at sec= %ld  nsec= %ld \n", time7.tv_sec, time7.tv_nsec);
	time_difference(&time7, &time6, &difference);
	syslog(LOG_INFO, "Difference = %ld  nsec= %ld \n", difference.tv_sec, difference.tv_nsec);	
	sem_post(&sem_frame);
	
 }	

 }// prediction function


int main( int argc, char** argv )
{

	center_count=0;
	frame_counter=0;
	cent_x_new= 0;
	cent_y_new= 0;
	cent_x_old= 0;
	cent_y_old= 0;


 //   cvNamedWindow("Capture Example", CV_WINDOW_AUTOSIZE);
    capture = (CvCapture *)cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);

    if(argc > 1)
    {
        sscanf(argv[1], "%d", &dev);
        printf("using %s\n", argv[1]);
    }
    else if(argc == 1)
        printf("using default\n");

    else
    {
        printf("usage: capture [dev]\n");
        exit(-1);
    }      


//---------------------Thread----------------

    int rt_max_prio, rt_min_prio;

    pthread_attr_init(&attr_frame);
    pthread_attr_init(&attr_center);
    pthread_attr_init(&main_attr);
    pthread_attr_init(&attr_pred);

    // initializing the mutex
    pthread_mutex_init(&lock,NULL);

	/* Configuring syslog */
	openlog("PROJECT_TRACK N POINT", 0, LOG_PID);

    pthread_attr_setinheritsched (&attr_frame, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy (&attr_frame, SCHED_FIFO);
    pthread_attr_setinheritsched (&attr_center, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy (&attr_center, SCHED_FIFO);
    pthread_attr_setinheritsched (&main_attr, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy (&main_attr, SCHED_FIFO);

pthread_attr_setinheritsched (&attr_pred, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy (&attr_pred, SCHED_FIFO);
   
    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    main_param.sched_priority = rt_max_prio;
    rt1_param.sched_priority = rt_max_prio-10;
    rt2_param.sched_priority = rt_max_prio-20;
    rt3_param.sched_priority = rt_min_prio;

    cpu_set_t cpuset;
    CPU_SET(0,&cpuset);
         pthread_attr_setaffinity_np(&main_attr, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_frame, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_center, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_pred, sizeof(cpu_set_t), &cpuset);


    // initialize the sequencer semaphores
    if (sem_init (&sem_frame, 0, 0)) { printf ("Failed to initialize sem_frame semaphore\n"); exit (-1); }
    if (sem_init (&sem_center, 0, 0)) { printf ("Failed to initialize sem_center semaphore\n"); exit (-1); }
    if (sem_init (&sem_pred, 0, 0)) { printf ("Failed to initialize sem_pred semaphore\n"); exit (-1); }


   
    pthread_attr_setschedparam (&attr_frame, &rt1_param);
    pthread_attr_setschedparam (&attr_center, &rt2_param);   
    pthread_attr_setschedparam (&main_attr, &main_param);
    pthread_attr_setschedparam (&attr_pred, &rt3_param);

int rc=0;



  /*Sem post forframe capture thread*/
	sem_post(&sem_frame);
     rc = pthread_create (&Thread_frame , &attr_frame , frame_capture, NULL);
    if(rc) printf("Failed to create thread 1\n");
     rc = pthread_create (&Thread_center , &attr_center , center, NULL);
    if(rc) printf("Failed to create thread 2\n");
     rc = pthread_create (&Thread_pred , &attr_pred , predict_position, NULL);
    if(rc) printf("Failed to create thread 1\n");

  /*Sem post forframe capture thread*/
	//sem_post(&sem_frame);

    pthread_join (Thread_frame , NULL );
    pthread_join ( Thread_center , NULL );
    pthread_join ( Thread_pred , NULL );
//------------------------------------------
   

    cvReleaseCapture(&capture);
    cvDestroyWindow("Capture Example");

closelog();

}; //main
