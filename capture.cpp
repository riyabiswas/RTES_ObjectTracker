
/* Headers section */
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <syslog.h>
#include <math.h>
#include <sys/param.h>
#include <sys/time.h>
#include <errno.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <syslog.h>


using namespace cv;
using namespace std;

#define HRES 640
#define VRES 400
#define OK 1
#define NSEC_PER_SEC 1000000000
//global variables
pthread_t  Thread_frame, Thread_center, Thread_pred, Thread_seq;
pthread_attr_t attr_frame, attr_center, attr_pred, main_attr, attr_seq; // make one attr for main
struct sched_param main_param, rt1_param, rt2_param, rt3_param, rt4_param; // make one param for main
CvCapture* capture;
IplImage* frame;
int dev=0;
pthread_mutex_t lock;
struct timespec time1, time2, time4, time5, time6, time7, cent_time[4], dif_time_old, dif_time_new, total_timeDif,difference;
double timeDif_ms_new, timeDif_ms_old, total_timeDif_ms;
char frame_captured[] = "Circle detection";
char Prediction_window[] ="Prediction window";


int cent_x[4];
int cent_y[4];

double speed_x_old;
double speed_y_old;

double speed_x_new;
double speed_y_new;

double acc_x;
double acc_y;

double dist_x_old;
double dist_y_old;

double dist_x_new;
double dist_y_new;

double nextPos_x;
double nextPos_y;

/*Define semaphores*/
sem_t sem_frame, sem_center, sem_pred;

int center_count;
int frame_counter;
int fps_count;
int cent_service_count;
int pred_service_count;

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
	stringstream pred_coordinates, x_value, y_value;

	int x=0;
	int y=0;
	int counter=0;

	Mat gray;
	vector<Vec3f> circles;
	clock_gettime(CLOCK_REALTIME, &time1);
	//syslog(LOG_INFO, "FREQUENCY: Center service at count 0: sec= %ld  nsec= %ld \n", time1.tv_sec, time1.tv_nsec);

	while(1)
	{   
		/*Hold semaphore*/
		sem_wait(&sem_center);

		clock_gettime(CLOCK_REALTIME, &time1);
		//syslog(LOG_INFO, "Center cal starts at sec= %ld  nsec= %ld \n", time1.tv_sec, time1.tv_nsec);

		/*Use mutex lock to protect global variable 'frame'*/
		pthread_mutex_lock(&lock);

		Mat mat_frame(frame);
		pthread_mutex_unlock(&lock);

		cvtColor(mat_frame, gray, CV_BGR2GRAY);
		GaussianBlur(gray, gray, Size(9,9), 2, 2);
		HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 50, 0, 0);


		syslog(LOG_DEBUG, "circles.size = %d\n", circles.size());

		for( size_t i = 0; i < circles.size(); i++ )
		{

			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			x= cvRound(circles[i][0]);
			y= cvRound(circles[i][1]);    

			// printing the center cordinates
			syslog(LOG_DEBUG,"Current centroid position X: %d, Y:%d\n", x,y);

			// circle center
			circle( mat_frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
			// circle outline
			circle( mat_frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}

		pthread_mutex_lock(&lock);

		for(int j=0;j<3;j++){
			// Saving the previous time stamp
			cent_time[j] =cent_time[j+1];

			// Saving the previoud cordinates of the center
			cent_x[j] = cent_x[j+1];
			cent_y[j] = cent_y[j+1];
			// Fetching the new cordinates of the center    

		}

		// Fetching the new timerepea
		clock_gettime(CLOCK_REALTIME, &cent_time[3]);
		//  syslog(LOG_DEBUG,"Time at which this centroid calculated =%ld second and %ld nanosecond\n", cent_time[3].tv_sec, cent_time[3].tv_nsec);

		cent_x[3] = x;
		cent_y[3] = y;

		pthread_mutex_unlock(&lock); //mutex lock over global variable released
   
		//io stream for put text
		(x_value << cent_x[3]);
		(y_value << cent_y[3]);
		pred_coordinates << "(" << x_value.str() << "," << y_value    .str() << ")";
		putText(mat_frame,pred_coordinates.str(),Point(cent_x[3],cent_y[3]),3,1,CV_RGB(0,255,0));

		// Clearing the IO stream	
		pred_coordinates.str(std::string());
		x_value.str(std::string());
		y_value.str(std::string());

		imshow(frame_captured, mat_frame);
		//frame_captured.setTo(Scalar(0,0,0));
		char c = cvWaitKey(10);              
		center_count = (center_count+1)%4; 

		cent_service_count++;


		clock_gettime(CLOCK_REALTIME, &time2);


		
		time_difference(&time2, &time1, &difference);
		



		
	}// while
}

/////////////////////////////////////////////////////////////////
void * frame_capture(void * a)
{
    
    
    
	//syslog(LOG_INFO, "FREQUENCY: fps at frame count 0: sec= %ld  nsec= %ld \n", time4.tv_sec, time4.tv_nsec);
	while(1){
			/*Hold semaphore for frame capture*/
			sem_wait(&sem_frame);

				clock_gettime(CLOCK_REALTIME, &time4);		//Get time for start of thread
				syslog(LOG_INFO, "frame cap start at sec= %ld  nsec= %ld \n", time4.tv_sec, time4.tv_nsec);

			pthread_mutex_lock(&lock);

			frame=cvQueryFrame(capture);
			if(!frame) break;  

			pthread_mutex_unlock(&lock);

			fps_count++;

			if(fps_count==3001)
			{
				exit(0);

			}


	}// while

} // frame capture funtion


/////////////////////////////////////////////////////////////////

//Fucntion predicting the next position of the centroid

////////////////////////////////////////////////////////////////



void * predict_position(void *threadid)
{

    stringstream pred_coordinates, x_value, y_value;
    timeDif_ms_new =0.0;
    timeDif_ms_old=0.0;
    total_timeDif_ms=0.0;
    speed_x_new=0;
    speed_y_new=0;
    acc_x=0;
    acc_y=0;
    Mat final_plot=Mat::zeros( 480, 640, CV_8UC3 );

    
    while(1)
     {
    
         /*Hold semaphore*/
            sem_wait(&sem_pred);
	
		clock_gettime(CLOCK_REALTIME, &time6);		//Get time for start of thread
		//syslog(LOG_INFO, "Pred starts at sec= %ld  nsec= %ld \n", time6.tv_sec, time6.tv_nsec);
	
        pthread_mutex_lock(&lock);
        dist_x_old = abs(cent_x[1]-cent_x[0]);			//Get distance travelled by centroid between C1 and C2, C3 and C4
        dist_y_old = abs(cent_y[1]-cent_y[0]);
        dist_x_new = abs(cent_x[3]-cent_x[2]);
        dist_y_new = abs(cent_y[3]-cent_y[2]);

        time_difference(&cent_time[1], &cent_time[0], &dif_time_old);
        time_difference(&cent_time[3], &cent_time[2], &dif_time_new);
        time_difference(&cent_time[3], &cent_time[0], &total_timeDif);

        timeDif_ms_new = (dif_time_new.tv_sec*1000000) + 34000000+dif_time_new.tv_nsec/1000; // in miliseconds, adding execution time for prediction service
        timeDif_ms_old = (dif_time_old.tv_sec*1000000) + 34000000+dif_time_old.tv_nsec/1000; // in miliseconds, adding execution time for prediction service
        total_timeDif_ms = (total_timeDif.tv_sec*1000000) + 34000000+total_timeDif.tv_nsec/1000; // in miliseconds, adding execution time for prediction service

        speed_x_new = dist_x_new/timeDif_ms_new;
        speed_y_new = dist_y_new/timeDif_ms_new;

        speed_x_old = dist_x_old/timeDif_ms_old;
        speed_y_old = dist_y_old/timeDif_ms_old;

        acc_x = (speed_x_new-speed_x_old)/total_timeDif_ms;
        acc_y = (speed_y_new-speed_y_old)/total_timeDif_ms;

        nextPos_x = cent_x[3] + speed_x_old*((double)total_timeDif_ms) + (0.5)*acc_x*((double)total_timeDif_ms)*((double)total_timeDif_ms);
        nextPos_y = cent_y[3] + speed_y_old*((double)total_timeDif_ms) + (0.5)*acc_y*((double)total_timeDif_ms)*((double)total_timeDif_ms);;
        pthread_mutex_unlock(&lock);
    
    //    final_plot.at<Vec3b>(nextPos_x,nextPos_y) = Vec3b(255, 150, 150);
         Point center(cvRound(nextPos_x), cvRound(nextPos_y));

	//io stream for put text
	(x_value << nextPos_x);
	(y_value << nextPos_y);
	pred_coordinates << "(" << x_value.str() << "," << y_value    .str() << ")";
	putText(final_plot,pred_coordinates.str(),Point(nextPos_x,nextPos_y),3,1,CV_RGB(0,255,0));
	// Clearing the IO stream	
	pred_coordinates.str(std::string());
	x_value.str(std::string());
	y_value.str(std::string());
	
        circle( final_plot, center, 3, Scalar(0,255,0), -1, 8, 0 );
        imshow(Prediction_window, final_plot);
// clear the screen                        
	final_plot.setTo(Scalar(0,0,0));
        char c = cvWaitKey(1);
    
         syslog(LOG_INFO, "Predicted centroid position X:%f, Y:%f \n\n\n",nextPos_x,nextPos_y);
        /*Release semaphore*/
         
        pred_service_count++;


     }    

 }// prediction function


/////////////////////////////////////////////////////////////////

//Function sequencing all the other threads

////////////////////////////////////////////////////////////////

void *Sequencer(void * unused)
{    
    while(1)
    {

	// The LCM of the system is 1000 ms
        usleep(99970); // period of thread_frame is 100 ms
clock_gettime(CLOCK_REALTIME, &time4);
       sem_post(&sem_frame);
       usleep(99970); // period of thread_center is 200 ms
clock_gettime(CLOCK_REALTIME, &time5);
time_difference(&time5, &time4, &difference);
syslog(LOG_INFO, "Frame request time = %ld  nsec= %ld \n", difference.tv_sec, difference.tv_nsec);
        sem_post(&sem_frame); sem_post(&sem_center);
        usleep(99970);
        sem_post(&sem_frame);
        usleep(99970);
        sem_post(&sem_frame); sem_post(&sem_center);
        usleep(99970);// period of thread_pred is 500 ms
        sem_post(&sem_frame); sem_post(&sem_pred);
        usleep(99970);
        sem_post(&sem_frame); sem_post(&sem_center);
        usleep(99970);
        sem_post(&sem_frame);
        usleep(99970);
        sem_post(&sem_frame); sem_post(&sem_center);
        usleep(99970);
        sem_post(&sem_frame);
        usleep(99970);
	// Posting all three threads at the same time
        sem_post(&sem_frame); sem_post(&sem_center); sem_post(&sem_pred);
    }
}


int main( int argc, char* argv[] )
{
    int rc=0;
    center_count=0;
    frame_counter=0;
    fps_count = 0;
    cent_service_count=0;
    pred_service_count=0;
    
    for(int j=0; j<4; j++)
	{

		cent_x[j]= 0;
		cent_y[j]= 0;
        
   	}

	/* Creating a window with AUTO_SIZE option for displaying the output */
    namedWindow( frame_captured, CV_WINDOW_AUTOSIZE );
    namedWindow( Prediction_window, CV_WINDOW_AUTOSIZE );


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

    int rt_max_prio, rt_min_prio;

    pthread_attr_init(&attr_frame);
    pthread_attr_init(&attr_center);
    pthread_attr_init(&main_attr);
    pthread_attr_init(&attr_pred);
    pthread_attr_init(&attr_seq);

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
    pthread_attr_setinheritsched (&attr_pred, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy (&attr_seq, SCHED_FIFO);

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

     rc = sched_getparam (getpid(), &main_param);
	
    main_param.sched_priority = rt_max_prio;
    rt1_param.sched_priority = rt_max_prio-20;
    rt2_param.sched_priority = rt_max_prio-30;
    rt3_param.sched_priority = rt_min_prio;
    rt4_param.sched_priority = rt_max_prio-10;
 
 
 /* Assigning main thread with scheduling policy FIFO and enable to act as sequencer */ 
	rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
	if (rc)
	{
		syslog(LOG_ERR, "sched_setscheduler rc is %d\n",rc);
		perror(NULL); exit(-1);
	}

    cpu_set_t cpuset;
    CPU_SET(0,&cpuset);
    pthread_attr_setaffinity_np(&main_attr, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_frame, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_center, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_pred, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr_seq, sizeof(cpu_set_t), &cpuset);


    // initialize the sequencer semaphores
    if (sem_init (&sem_frame, 0, 1)) 
     { printf ("Failed to initialize sem_frame semaphore\n"); exit (-1); }
   if (sem_init (&sem_center, 0, 1)) 
     { printf ("Failed to initialize sem_center semaphore\n"); exit (-1); }
    if (sem_init (&sem_pred, 0, 1)) 
    { printf ("Failed to initialize sem_pred semaphore\n"); exit (-1); }




    pthread_attr_setschedparam (&attr_frame, &rt1_param);
    pthread_attr_setschedparam (&attr_center, &rt2_param);   
    pthread_attr_setschedparam (&main_attr, &main_param);
    pthread_attr_setschedparam (&attr_pred, &rt3_param);
    pthread_attr_setschedparam (&attr_seq, &rt4_param);

 
     rc = pthread_create (&Thread_frame , &attr_frame , frame_capture, NULL);
 	    if(rc) printf("Failed to create thread 1\n");
	
     rc = pthread_create (&Thread_center , &attr_center , center, NULL);
 	   if(rc) printf("Failed to create thread 2\n");
	
     rc = pthread_create (&Thread_pred , &attr_pred , predict_position, NULL);
 	   if(rc) printf("Failed to create thread 1\n");

     rc = pthread_create (&Thread_seq , &attr_seq , Sequencer, NULL);
 	   if(rc) printf("Failed to create thread 1\n");


    pthread_join (Thread_seq , NULL );
    pthread_join (Thread_frame , NULL );
    pthread_join ( Thread_center , NULL );
    pthread_join ( Thread_pred , NULL );

   

   	 cvReleaseCapture(&capture);
	if(pthread_attr_destroy(&attr_frame) != 0)
		perror("attr destroy");
	
	if(pthread_attr_destroy(&attr_center) != 0)
		perror("attr destroy");
	
	if(pthread_attr_destroy(&attr_pred) != 0)
		perror("attr destroy");
	
	
	/* Stop capturing and free the resources */
        cvReleaseCapture(&capture);
	
	sem_destroy(&sem_frame);
	sem_destroy(&sem_center);
	sem_destroy(&sem_pred);

	if(pthread_mutex_destroy(&lock) != 0)
		perror("mutex lock destroy");

   	closelog();

}; //main
