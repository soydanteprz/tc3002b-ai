import java.util.*;
import java.lang.*;
import java.io.*;

public class Main
{
	static PrintWriter out;
	static FastReader scan;

    /*-------- I/O using short named function ---------*/
	
	public static int ni(){return scan.nextInt();}
	public static long nl(){return scan.nextLong();}
	
	
	
	
    /*-------- for output of an array ---------------------*/
     
	
	
	
    /*-------------- for input in an array ---------------------*/
	
	static void lIA(long arr[]){
	     for(int i=0; i<arr.length; i++)arr[i] = nl(); 
	}
	
	
	/*------------ for taking input faster ----------------*/
     

	// Method to check if x is power of 2
    
    //Method to return lcm of two numbers
    
    //Method to count digit of a number
    

    //Method for sorting
	
	//Method for checking if a number is prime or not
	 

	public static void main (String[] args) throws java.lang.Exception
	{
		OutputStream outputStream =System.out;
        out =new PrintWriter(outputStream);
        scan =new FastReader();
		//for fast output sometimes
		StringBuilder sb = new StringBuilder();
		int t = ni();
		while(t-->0){
			int n = ni();
			long arr[] = new long[n];
			lIA(arr);
			long ans = (long)(n*(arr[0] + arr[1]));
			long sum = arr[0] + arr[1];
			long emin = arr[0], omin = arr[1];
			
			for(int i=2; i<n; i++){
				sum += arr[i];
				if(i%2==0){
					emin = Math.min(arr[i], emin);
				}
				else{
					omin = Math.min(arr[i], omin);
				}
				long temp = sum - emin - omin;
				if(i%2==0)
					temp += (n-i/2)*emin + (n-i/2+1)*omin;
				else
					temp += (n-(i-1)/2)*(emin + omin);
				ans = Math.min(ans, temp);

			}
			System.out.println(ans);
        }
        out.flush();
		out.close();
	}
	
}
