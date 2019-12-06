
import java.text.DecimalFormat;
import java.util.Random;

public class jvpyMonze {
	private static int numInputs= 4;
	private static int numHidden= 6;
	private static int numOutput= 14;
	private static double rRate = 0.5;	

	private static int trainingCycles=10000;
	private static double wIh[][] = new double[numInputs + 1][numHidden];
	//weights comming into hidden layer
	private static double whO[][] = new double[numHidden + 1][numOutput];	
	//weights going out of hidden layer
	private static double[] inputs = new double[numInputs]; // input neuron array
	private static double[] hidden=new double[numHidden];	// hidden neuron array
	private static double[] expected=new double[numOutput];	// expected output array
	private static double[] predicted=new double[numOutput];// actuall predicted array
	
	private static double[] errForOP=new double[numOutput]; // error of output
	private static double[] errForH=new double[numHidden];  // error of hidden layer
	private static int samples=7;
	
	private static double trainInputs[][] = new double[][] {
		{1, 0, 0, 1},  // 1
		{0, 1, 0, 1},  // 2
		{1, 1, 1, 0},  // 3
		{0, 0, 0, 1},  // 4
		{1, 0, 1, 1},  // 5
		{0, 1, 1, 1},  // 6
		{1, 1, 1, 1}}; // 7	
	private static int trainOutput[][] = new int[][] 
	   {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 1
	    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 2
	    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 3
	    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},  // 4
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},  // 5
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},  // 6
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}}; // 7  Perfect Health Results
													
	public static void main(String[] args) {
		int currSample=0;
		assignRandWeights();
		for(int epoch=0;epoch<trainingCycles;epoch++){currSample++;
			if(currSample==samples){currSample=0;}
			for(int i=0;i<numInputs;i++){inputs[i]=trainInputs[currSample][i];}
			for(int i=0;i<numOutput;i++){expected[i]=trainOutput[currSample][i];	}
			feedForward();	backProp();	}

		System.out.println("\n================= START ====================== ");
		System.out.println("\nTest network against lab input: LaborotoryX");
		testNetwork(trainInputs); 		getAccuracy(trainInputs,trainOutput);    
        System.out.println("\nTest network against noisy input: SmartHouseX");
		double[][] patientA = getMedicalsX();
		testNetwork(patientA);  	     getAccuracy(patientA, trainOutput);
		System.out.println("\n===== If less then 80% then Danger Alarm ===== "); 
		System.out.println("\n=================== END ====================== ");   
	}
	//=====================================================================================0
	//Get the accuracy for inputs when the expected output is given
	private static void getAccuracy(double[][] input,	int[][] output){
		double sum=0.0;
		for(int i=0;i<samples;i++){
			for(int j=0;j<numInputs;j++){ inputs[j]  =input[i][j];}
			for(int j=0;j<numOutput;j++){ expected[j]=output[i][j];}
			feedForward();

			if(maximumIdx(predicted)==maximumIdx(expected)){sum++;}
			else{
				System.out.println("============ The Defective Line ==============");
                System.out.println(inputs[0] + "\t" + inputs[1] + "\t" + inputs[2] + "\t" + inputs[3]);
				System.out.println("=========== Predicted_Max Vs Expected_Max ===========");
				System.out.println(maximumIdx(predicted) + "\t" + maximumIdx(expected));
              }
		}
		System.out.println("Medical Diagnosis Accuracy for training "+ sum*100/samples);
	}
	//===================================================================================1
	private static double[][] getMedicalsX(){
	double patient1[][] = new double[][] {
	{1, 1, 1, 0},  //1
	{0, 1, 1, 0},  //2
	{1, 0, 0, 1},  //3
	{0, 0, 0, 1},  //4
	{0, 0, 1, 1},  //5
	{0, 1, 1, 1},  //6
	{1, 1, 1, 1}}; //7
	return patient1;
	}
	// ==================================================================================2
	//test network is testing test data for expected and predicted
	private static void testNetwork(double[][] testInputs){
		DecimalFormat dfm = new java.text.DecimalFormat("###0.0");
		System.out.println();
		for(int i=0;i<testInputs.length;i++){
			for(int j=0;j<numInputs;j++){inputs[j]=testInputs[i][j];}
			feedForward();
			for(int j = 0; j < numInputs; j++)
            {System.out.print(dfm.format(inputs[j]) + "\t"); } // j
				System.out.print("Predicted: " + maximumIdx(predicted) +
				"-Expected:"+maximumIdx(expected) + "\n");
		}
	}
	//=================================================================================3
	private static void assignRandWeights(){
		for(int i=0;i<=numInputs;i++){
			for(int j=0;j<numHidden;j++){wIh[i][j]=(new Random().nextDouble())-0.5;}}
		for(int i=0;i<=numHidden;i++){
			for(int j=0;j<numOutput;j++){whO[i][j]=(new Random().nextDouble())-0.5;}}
	}
	//================================================================================= 4
	//This is acctually the predicting part. Depending on the weights of the neurons
	//it tries to come up with a output that is the best match for given inputs.
	private static void feedForward(){
		double sum=0.0;
		for(int i=0;i<numHidden;i++){sum=0.0;
			for(int j=0;j<numInputs;j++){sum+=wIh[j][i]*inputs[j];}
			sum+=wIh[numInputs][i];
			hidden[i]=sigmoid(sum);	}
		for(int i=0;i<numOutput;i++){	sum=0.0;
			for(int j=0;j<numHidden;j++){sum+=hidden[j]*whO[j][i];}
			sum+=whO[numHidden][i];
			predicted[i]=sigmoid(sum);}
	}
	//================================================================================= 5
	//Back prop is similar to learning from mistakes. We adjust the neuron weights depending on the
	//difference between the predicted and expected outputs.
	private static void backProp(){
		for(int i=0;i<numOutput;i++){
			errForOP[i]=(expected[i]-predicted[i])*sigmoidDerivative(predicted[i]);}
		for(int i=0;i<numHidden;i++){	errForH[i]=0.0;
			for(int j=0;j<numOutput;j++){errForH[i]+=errForOP[j]*whO[i][j];}
			errForH[i]*=sigmoidDerivative(hidden[i]);}
		for(int i=0;i<numOutput;i++){
			for(int j=0;j<numHidden;j++){whO[j][i]+=(rRate * errForOP[i] * hidden[j]);}
			whO[numHidden][i] += (rRate * errForOP[i]);	}
		for (int i = 0; i < numHidden; i++) {
			for (int j = 0; j < numInputs; j++) {wIh[j][i] += (rRate * errForH[i] * inputs[j]);}
			wIh[numInputs][i] += (rRate * errForH[i]);
		}
	}
	
	//	this is called the sigmoid function
	private static double sigmoid(double x){ return 1.0/(1.0+Math.exp(-x));	}
	//this is the derivate of the sigmoid function
	private static double sigmoidDerivative(double x){return x*(1.0-x);	}
	
	//================================================================================= 6
	private static int maximumIdx(final double[] vector)
    {  //  This function returns the index of the maximum of vector().
        int Vmax = 0;
        double max = vector[Vmax];
        for(int index = 0; index < numOutput; index++)
		{if(vector[index] > max){ max = vector[index]; Vmax = index; }}
	    return Vmax;
    }
}
     //================================================================================= 7