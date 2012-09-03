#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include"nn.h"
#include<time.h>

#define BUFFERSIZE 2000

// Global variables
int maxLevels=0;
int *layerSizeList;
float learningRate=0;
int iterations;
char *trainingFileName;
int numberOfFacts=0;
int reportFrequency=0;
int reportW=0;

double *AllocateVector(long length){
  double *v; 
  v=(double *)malloc((size_t) ((length + 1)*sizeof(double)));
  if (!v) printf("allocateVector: failed");
  return v;
}

unsigned char *AllocateCvector(long length){
  unsigned char *v;
  v=(unsigned char *)malloc((size_t) ((length + 2)*sizeof(unsigned char)));
  if (!v) printf("allocateCvector: failed");
  return v+1;
}

int *AllocateIvector(long length){
  int *v;
  v=(int *)malloc((size_t) ((length + 1)*sizeof(int)));
  if (!v) printf("allocateCvector: failed");
  return v;
}

double **AllocateMatrix(long nrows, long ncols){
  long i;
  double **m;

  m=(double **) malloc((size_t)((nrows+1)*sizeof(double*)));
  if (!m) printf("allocatedMatrix: failure");

  m[1]=(double *) malloc((size_t)((nrows*ncols+1)*sizeof(double)));
  if (!m[1]) printf("allocateMatrix: failure");

  for(i=2;i<=nrows;i++) m[i]=m[i-1]+ncols;

  return m;
}

double Frand(){
  double x;
  x = (rand() / ((double)RAND_MAX + 1));
  return x;
}

double RandomGaussian() {
    double u1, u2;  
    double w;       
    double g1, g2;  

    do {
        u1 = 2 * Frand() - 1;
        u2 = 2 * Frand() - 1;
        w = u1*u1 + u2*u2;
    } while ( w >= 1 );

    w = sqrt( (-2 * log(w))  / w );
    g1 = u2 * w;
    return g1;
}

void FinalReport(double ***w, double **trainingInput, double **trainingOutput){
  printf("\nFinal Report: \n");
  printf("Iterations %d Final Score %e\n", iterations, Score(w, trainingInput, trainingOutput));
  TestTrainingSet(w,trainingInput,trainingOutput);
}

void TestTrainingSet(double ***w, double **trainingInput, double **trainingOutput){ 
  int i,j;
  double **states;

  states = AllocateState(layerSizeList);

  for(i=1;i<=numberOfFacts;i++){
    PropagateNN(w, trainingInput[i], states);
    printf("Fact %d : Predicted : ",i);
    for(j=1;j<=layerSizeList[maxLevels];j++) printf("%0.3f ",states[maxLevels][j]);
    printf("True : ");
    for(j=1;j<=layerSizeList[maxLevels];j++) printf("%0.3f ",trainingOutput[i][j]);
    printf("\n");
  }
}

void ReadConfigFile(char *filename){
  FILE *ifp;
  char layerSizeListString[BUFFERSIZE]; 
  char trainingFileNameString[BUFFERSIZE]; 
  char buffer[BUFFERSIZE];
  int i;
  
  ifp = fopen(filename, "r");

  while ( fgets(buffer, sizeof buffer, ifp) ){
    if ( buffer[0] == '#' ){
      continue;
    }
    if ( sscanf(buffer, "MAX_LEVELS %d", &maxLevels) == 1 ){
      continue;
    }
    if ( sscanf(buffer, "STATE_SIZE_LIST %[0-9 ]s",layerSizeListString) == 1 ){
      continue;
    }
    if ( sscanf(buffer, "LEARNING_RATE %f",&learningRate) == 1 ){
      continue;
    }
    if ( sscanf(buffer, "ITERATIONS %d", &iterations) == 1 ){
      continue;
    }
    if ( sscanf(buffer, "INPUT_FILE_NAME %s", trainingFileNameString) == 1 ){
      continue;
    }
    if ( sscanf(buffer, "REPORT_FREQUENCY %d", &reportFrequency) == 1 ){
      continue;
    }
    if ( sscanf(buffer, "REPORT_W%d", &reportW) == 1 ){
      continue;
    }
  }

  fclose(ifp);

  // Do some processing here to get the layerSizeList since we have to wait for maxLevels
  trainingFileName = AllocateCvector(strlen(trainingFileNameString));
  strcpy(trainingFileName, trainingFileNameString);
  layerSizeList = AllocateIvector(maxLevels);
  layerSizeList[1] = atoi(strtok(layerSizeListString," "));
  for(i=2;i<=maxLevels;i++){
    layerSizeList[i] = atoi(strtok(NULL," "));
  }

  // Count lines in the training file for number of facts
  ifp = fopen(trainingFileName, "r");
  fgets(buffer,2000,ifp);  // get rid of header line
  while(fgets(buffer,2000,ifp)){
    numberOfFacts++;
  }
  return;
}

void ImportTrainingSet(char *filename, double **inputTrainingSet, double **outputTrainingSet, int *layerSizeList){
  FILE *ifp;
  char line[2000];
  int i=0,j=0, numberOfTrainingPoints=0;
  char tmp[200];
  double **m;

  if(!(ifp=fopen(filename,"r"))){
    printf("import: can't open %s.\n",filename);
    exit(1);
  }
  
  fgets(line,2000,ifp); // get rid of header line
  
  while(fgets(line,2000,ifp)){
    if(line[0] == '#'){continue;} // skip comments
    strtok(line," "); // remove first element which is only an index or label for the fact number
    i++; 
    for(j=1;j<=layerSizeList[1];j++){
      strcpy(tmp,strtok(NULL, " ")); 
      inputTrainingSet[i][j]=atof(tmp);
    }
    for(j=1;j<=layerSizeList[maxLevels];j++){
      strcpy(tmp,strtok(NULL, " ")); 
      outputTrainingSet[i][j]=atof(tmp);
    }
  }
}
 
double Sigmoid(double x){
  return 2 / (1 + exp(-x)) - 1;
}

double SigmoidToDsigmoid(double x){
  return 1.0/2.0 * (1 + x) * (1 - x);
}

void PopulateW(double ***w){
  int i,j,k;

  for(i=1;i<maxLevels;i++){
    for(j=1;j<=layerSizeList[i+1];j++){
      for(k=1;k<=layerSizeList[i]+1;k++){
	w[i][j][k] = RandomGaussian();
      }
    }
  }
}

void PopulateStates(double **states, int *layerSizeList){
  int i,j;

  for(i=1;i<=maxLevels;i++){
    for(j=1;j<=layerSizeList[i];j++){
      states[i][j] = RandomGaussian();
    }
  }
}
	
void PropagateNN(double ***w, double *inputState, double **states){
  int i,j,k;

  CopyVector(inputState, states[1], layerSizeList[1]);

  for(i=1;i<maxLevels;i++){
    for(j=1;j<=layerSizeList[i+1];j++){
      states[i+1][j] = w[i][j][1];  // Offset or constant field element
      for(k=1;k<=layerSizeList[i];k++){
	states[i+1][j] += w[i][j][k+1] * states[i][k]; // First element of a row in w is the offset, so we must start multiplying from 2
      }
      states[i+1][j] = Sigmoid(states[i+1][j]);
    }
  }
}

void PrintStates(double **states, int *layerSizeList){
  int i,j;

  printf("Input Layer:\n");
  for(i=1;i<=layerSizeList[1];i++){
    printf("%.2e ",states[1][i]);
  }
  printf("\nIntermediate Layers:");
  for(i=2;i<maxLevels;i++){
    printf("\n");
    for(j=1;j<=layerSizeList[i];j++){
      printf("%.2e ",states[i][j]);
    }
  }
  printf("\nOutput Layer:\n");
  for(i=1;i<=layerSizeList[maxLevels];i++){
    printf("%.2e ",states[maxLevels][i]);
  }
  printf("\n");
  return;
}

double Score(double ***w, double **inputStates, double **targetStates){
  int i,j;
  double **currentStates, rms=0;

  currentStates = AllocateState(layerSizeList);
  
  for(i=1;i<=numberOfFacts;i++){
    PropagateNN(w, inputStates[i], currentStates);
    for(j=1;j<=layerSizeList[maxLevels];j++){
      rms += (targetStates[i][j] - currentStates[maxLevels][j])*(targetStates[i][j] - currentStates[maxLevels][j]);
    }
  }
  return ((float)rms) / numberOfFacts;
}

void PrintW(double ***w, int *layerSizeList){
  int i,j,k;

  for(i=1;i<maxLevels;i++){
    printf("\nw[%d]\n\t",i);
    for(j=1;j<=layerSizeList[i+1];j++){
      for(k=1;k<=layerSizeList[i]+1;k++){
	printf("%.2e ",w[i][j][k]); 
      }
      printf("\n\t");
    }
  }
  printf("\n");
  return;
}

double **AllocateState(int *layerSizeList){
  int i;
  double **s;
  
  s = (double **) malloc( sizeof(double *) * (maxLevels + 1));
  for(i=1;i<=maxLevels;i++){
    s[i] = AllocateVector(layerSizeList[i]);
  }
  return s;
}

double ***AllocateW(int *layerSizeList){
  int i;
  double ***w;

  w = (double ***) malloc( sizeof(double **) * (maxLevels + 1));
  for(i=1;i<maxLevels;i++){
    w[i] = AllocateMatrix(layerSizeList[i+1],layerSizeList[i]+1); 
  }
  return w;
}

void CopyVector(double *source, double *target, int size){
  int i;
  for(i=1;i<=size;i++){
    target[i] = source[i]; 
  }
  return;
}

void copyMatrixRowVector(double **source, int index, double *target, int size){
  int i;
  for(i=1;i<=size;i++){
    target[i] = source[index][i]; 
  }
  return;
}

void GetFirstError(double *currError, double *currOutput, double *targetOutput, int size){
  int i;

  for(i=1;i<=size;i++){
    currError[i] = -1 * (currOutput[i] - targetOutput[i]) * SigmoidToDsigmoid( currOutput[i] );
  }
  return;
}

void BackPropagate(double **targetInputSet, double **targetOutputSet, double ***w, int *layerSizeList, int iter){
  double **error;
  double **nextError;
  double **currentStates;
  double *currOutput;
  double **currError;
  double *targetInput;
  double *targetOutput;
  double ***dw;
  int i,j,k,m,loop;

  dw = AllocateW(layerSizeList);

  currentStates = AllocateState(layerSizeList);
  error = AllocateState(layerSizeList);
  nextError = AllocateState(layerSizeList);
  currError = AllocateState(layerSizeList);

  currOutput = AllocateVector(layerSizeList[maxLevels]);
  targetOutput = AllocateVector(layerSizeList[maxLevels]);
  targetInput = AllocateVector(layerSizeList[1]);

  for(loop=1;loop<=iter;loop++){

    if((reportFrequency!=0) && (loop % reportFrequency)==0){
      printf("Step %d - score %e\n",loop, Score(w,targetInputSet,targetOutputSet));
    }
    // clear dw
    for(i=maxLevels-1;i>=1;i--){
      for(j=1;j<=layerSizeList[i+1];j++){
        for(k=1;k<=layerSizeList[i]+1;k++){
	  dw[i][j][k] = 0;
	}
      }
    }

    for(m=1;m<=numberOfFacts;m++){
      //CopyVector(targetInput, currentStates[1], layerSizeList[1]);
      copyMatrixRowVector(targetInputSet,m,targetInput,layerSizeList[1]);
      copyMatrixRowVector(targetOutputSet,m,targetOutput,layerSizeList[maxLevels]);

      // Set currentStates layer 1 to the input
      PropagateNN(w, targetInput, currentStates); 

      // Calculating all error states first
      // Calculating the first error state at the final layer
      GetFirstError(error[maxLevels], currentStates[maxLevels], targetOutput, layerSizeList[maxLevels]);
      // Screw up mismatch between currError here and in Mathematica
      for(i=maxLevels-1;i>=1;i--){
        for(j=1;j<=layerSizeList[i];j++){
          error[i][j] = 0;
          for(k=1;k<=layerSizeList[i+1];k++){
            error[i][j] += w[i][k][j+1] * error[i+1][k];
          } 
        }
        // SigmoidToDsigmoid of currentStates
        for(j=1;j<=layerSizeList[i];j++){
          error[i][j] = error[i][j] * SigmoidToDsigmoid(currentStates[i][j]);
        }
      }

      for(i=maxLevels-1;i>=1;i--){
        for(j=1;j<=layerSizeList[i+1];j++){
          dw[i][j][1] += error[i+1][j] * learningRate;
          for(k=1;k<=layerSizeList[i];k++){
            dw[i][j][k+1] += error[i+1][j] * currentStates[i][k] * learningRate;
          }
        }
      }
    }

    for(i=maxLevels-1;i>=1;i--){
      for(j=1;j<=layerSizeList[i+1];j++){
        for(k=1;k<=layerSizeList[i]+1;k++){
	  w[i][j][k] += dw[i][j][k];
	}
      }
    }
  }
  return;
}

void ImportW(char *filename, double ***w, int *layerSizeList){
  int i,j,k;
  char line[12000];
  FILE *ifp;

  if(!(ifp=fopen(filename,"r"))){
    printf("import: can't open %s.\n",filename);
    exit(1);
  }

  for(i=1;i<maxLevels;i++){
    for(j=1;j<=layerSizeList[i+1];j++){
      fgets(line,12000,ifp);
      w[i][j][1]=atof(strtok(line," "));
      for(k=2;k<=layerSizeList[i]+1;k++){
        w[i][j][k]=atof(strtok(NULL," "));
      }
    }
  }
  return;
}
	

void ExportW(char *filename, double ***w, int *layerSizeList){
  int i,j,k;
  char line[2000];
  FILE *ofp;

  if(!(ofp=fopen(filename,"w"))){
    printf("import: can't open %s.\n",filename);
    exit(1);
  }

  for(i=1;i<maxLevels;i++){
    for(j=1;j<=layerSizeList[i+1];j++){
      fprintf(ofp,"%.5e",w[i][j][1]);
      for(k=2;k<=layerSizeList[i]+1;k++){
        fprintf(ofp," %.5e",w[i][j][k]);
      }
      fprintf(ofp,"\n");
    }
  }
  return;
}

int main(int argc, char *argv[]){
  double **m;
  double **trainingInput, **trainingOutput;
  int i,j;
  double **states, ***w, ***dw, avg, sigma;

  srand(time(NULL));

  ReadConfigFile("nn.config");
  trainingInput=AllocateMatrix(numberOfFacts,layerSizeList[1]);
  for(i=1;i<=maxLevels;i++){
  }
  trainingOutput=AllocateMatrix(numberOfFacts,layerSizeList[maxLevels]);
  m=AllocateMatrix(layerSizeList[1],numberOfFacts);
  ImportTrainingSet(trainingFileName, trainingInput, trainingOutput, layerSizeList);

  // Allocate states and w matrices

  states = AllocateState(layerSizeList);
  w = AllocateW(layerSizeList);
  dw = AllocateW(layerSizeList);

  PopulateW(w);  
  PopulateStates(states, layerSizeList);

  BackPropagate(trainingInput, trainingOutput, w, layerSizeList, iterations);

  FinalReport(w, trainingInput, trainingOutput);

  if(reportW){ExportW("output-W.out", w, layerSizeList);}
} 
