import java.util.*;
import java.lang.*;
import java.io.*;


public class cf {
    static PrintWriter out;
    static FastReader scan;

    /*-------- I/O using short named function ---------*/
    

    public static int ni() {
        return scan.nextInt();
    }

    

    

    

    

    

    

    /*-------- for output of an array ---------------------*/
    

    static void lPA(long arr[]) {
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < arr.length; i++) output.append(arr[i] + " ");
        out.println(output);
    }

    

    

    /*-------------- for input in an array ---------------------*/
    static void iIA(int arr[]) {
        for (int i = 0; i < arr.length; i++) arr[i] = ni();
    }

    

    

    

    /*------------ for taking input faster ----------------*/
    

    

    // Method to check if x is power of 2
    

    //Method to return lcm of two numbers
    

    //Method to count digit of a number
    

    

    static final Random random = new Random();

    //Method for sorting
    

    

    //Method for checking if a number is prime or not
    

    public static void main(String[] args) throws java.lang.Exception {
        OutputStream outputStream = System.out;
        out = new PrintWriter(outputStream);
        scan = new FastReader();
        //for fast output sometimes
        StringBuilder sb = new StringBuilder();
        int t = ni();
        while (t-- != 0) {
            int n=ni();
            int k=ni();
            int[] a=new int[k];
            int[] temp=new int[k];
            iIA(a);
            iIA(temp);
            long dp[]=new long[n];
            Arrays.fill(dp,Integer.MAX_VALUE);
            for(int i=0;i<k;i++){
                dp[a[i]-1]=temp[i];
            }
            //iPA(dp);
            for(int i=1;i<n;i++){
                dp[i]=Math.min(dp[i-1]+1,dp[i]);
            }
            //iPA(dp);
            for(int i=n-2;i>=0;i--){
                dp[i]=Math.min(dp[i+1]+1,dp[i]);
            }
            lPA(dp);
            //pn("");
        }

        out.flush();
        out.close();
    }
}
