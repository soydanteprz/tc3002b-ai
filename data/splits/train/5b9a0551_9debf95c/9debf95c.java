import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;


public class Main {


    static class Clock {

        protected long start, stop;

        public void start() {
            start = System.currentTimeMillis();
        }

        

        
    }


    public static PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));

    


    


    static long mod = 1000000007;

    

    


    

    

    static int parent[];
    static int rank[];

    

    


    public static long arr[];
    public static int arr1[];

//    static void seive(int n) {
//        arr = new int[n + 1];
//        arr[0] = arr[1] = 1;
//        for (int i = 4; i <= n; i = i + 2) {
//            arr[i] = 1;
//        }
//        for (int i = 3; i * i <= n; i = i + 2) {
//            if (arr[i] == 0) {
//                for (int j = i * i; j <= n; j = j + i) {
//                    arr[j] = 1;
//                }
//
//            }
//        }
//    }

    

    
    
    

    public static void main(String[] args) throws Exception {
        FastReader sc = new FastReader();

        Clock clock = new Clock();
        clock.start();
       int t1=sc.nextInt();
        for (int t = 0; t <t1 ; t++) {
          int n=sc.nextInt();
          long arr[]=new long[n];
            for (int i = 0; i < n; i++) {
                arr[i]=sc.nextLong();
            }
          long sum1=0;
          long sum2=0;
          long min1 =Long.MAX_VALUE;
          long min2 =Long.MAX_VALUE;
          long brr[]=new long[n];
            for (int i = 0; i <n ; i+=2) {

                int nn =i/2;
                min1=Math.min(arr[i],min1);
                sum1+=arr[i];
                long x =(sum1-min1) + min1*(n-nn);
               // System.out.println(x);
               brr[i]=x;

            }
            for (int i = 1; i <n ; i+=2) {
                int nn =i/2;

                //System.out.println(x);
                min2=Math.min(arr[i],min2);
                sum2+=arr[i];
                long x =(sum2-min2) + min2*(n-nn);
                brr[i]=x;
            }
            long min=Long.MAX_VALUE;
            for (int i = 1; i < n; i++) {
                min=Math.min(brr[i]+brr[i-1],min);
            }
            out.println(min);
            //out.println(Arrays.toString(brr));

           // System.out.println(min1+" "+min2);

        }
        out.close();


    }


}







