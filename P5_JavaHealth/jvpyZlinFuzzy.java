import java.util.Scanner;
public class jvpyZlinFuzzy {

	private static double minrange = 0.0, maxrange =100.0;
	private static double[] t = new double[100];  // Tested array
	private static double[] p = new double[100];  // Predicted array
	private static double[] wt = new double[100];
	public static int i = 3;
	public static int j = 0;
	
	public static void main(String[] args) {

		Scanner s = new Scanner(System.in);
		System.out.println("Enter the initial weight value(-0.09 to +0.09)");
		double temp = s.nextDouble();
		wt[0] = wt[1] = wt[2] = wt[3] = wt[4] = wt[5] = wt[6] = wt[7] 
		= wt[8] = wt[9] = wt[10]=wt[11]=temp;
		System.out.println("Records of the latest four temperature readings");

		t[0] = s.nextDouble(); 		// Type first value
		t[1] = s.nextDouble();		// Type second value
		t[2] = s.nextDouble();  	// Type third value
		while (i != 30) {
			t[i] = s.nextDouble();  // T[3]
			i++;
		System.out.println("Previous Temperature Outputs are :"+i+":["+t[i - 4]+ " "+
		t[i - 3]+ " "+ t[i - 2]+" "+ t[i - 1]+"]");
		p[j++] = predict(t[i - 4], t[i - 3], t[i - 2], t[i - 1])+(t[i-1]+t[i-2]+t[i-3]+t[i-4])/4;
		System.out.println("After " + i +" Minutes, the ANN controlled-temperature is :"	
		+ p[j - 1]);
		System.out.println("What is your desired Temperature?");
		}
	}

	private static double[] updatewt( double a, double b, double c,
			double d, double e, double f, double o) {
		if(o>=0.1){
		wt[0] =normalize( wt[0] - a*b*c*d);
		wt[1] =normalize( wt[1] - b*c*d*e);
		wt[2] =normalize( wt[2] - c*d*e*f);
		wt[3] =normalize( wt[3] - d*e*f*a);
		wt[4] =normalize( wt[4] - e*f*a*b);
		wt[5] =normalize( wt[5] - f*a*b*c);
		wt[6] =normalize( wt[6]- (p[j]-t[i])/2);
		wt[7] =normalize( wt[7]-(p[j-1]-t[i-1])/2);
		wt[8] =normalize( wt[8]-(p[j-2]-t[i-2])/2);
		wt[9] =normalize( wt[9]-(p[j-3]-t[i-3])/2);
		wt[10] =normalize( wt[10]-(p[j-4]-t[i-4])/2);
		wt[11] =normalize( wt[11]-(p[j-5]-t[i-5])/2);}
		return wt;
	}

	private static double predict(double a, double b, double c, double d) {
		// Neurons in Input Layer #1
		double p1 = normalize(a);
		double p2 = normalize(b);
		double p3 = normalize(c);
		double p4 = normalize(d);

		// Neurons in Hidden Layer #2
		double n1 = normalize((p1 * wt[0] + p2 * wt[1] + p3 * wt[2] + p4* wt[3]));
		double n2 = normalize((p1 * wt[1] + p2 * wt[2] + p3 * wt[3] + p4* wt[4]));
		double n3 = normalize((p1 * wt[2] + p2 * wt[3] + p3 * wt[4] + p4* wt[5]));
		double n4 = normalize((p1 * wt[3] + p2 * wt[4] + p3 * wt[5] + p4* wt[0]));
		double n5 = normalize((p1 * wt[4] + p2 * wt[5] + p3 * wt[0] + p4* wt[1]));
		double n6 = normalize((p1 * wt[5] + p2 * wt[0] + p3 * wt[1] + p4* wt[2]));
	//	System.out.println("Intermediate neuronX values:["+n1+"  "+n2+"  "+n3+"  "+n4+"  "+n5+"  "+n6);
		// One Neuron in the output Layer #3
		double out = normalize(n1*wt[6]+n2*wt[7]+n3*wt[8]+n4*wt[9]+n5*wt[10]+n6*wt[11]);
		wt = updatewt(n1, n2, n3, n4, n5, n6, out);
		return denormalize(out)*100;
	}

	private static double denormalize(double out) {	return  (out * maxrange);	}
	private static double normalize(double d) {
	//System.out.println("norm "+ ((double)(maxrange - d)) / ((double)(maxrange - minrange)));
		double temp=1.0-((double)(maxrange) - (d)) / ((double)(maxrange - minrange));
		if(temp>1.0){temp=normalize(temp);}
		return temp;
	}
}
