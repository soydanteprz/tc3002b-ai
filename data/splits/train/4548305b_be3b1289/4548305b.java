import java.util.function.Consumer;
import java.util.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.io.*;
import java.lang.Math.*;

 public class KickStart2020{
	
	
	
	
	 
	
	
	
	
	 
	 
	 
	 
		public static void main(String[] args) throws Exception{
			FastReader sc = new FastReader();
			PrintWriter out = new PrintWriter(System.out);
          int t = sc.nextInt();
          while(t-- > 0) {
           int n = sc.nextInt();
           int k = sc.nextInt();
           int arr[] = new int[k];
           int temp[] = new int[k];
           for(int i = 0; i < k; i++) arr[i] = sc.nextInt();
           for(int i = 0; i < k; i++) temp[i] = sc.nextInt();
           long brr[] = new long[n];
           Arrays.fill(brr, Integer.MAX_VALUE);
           for(int i = 0; i < k; i++) brr[arr[i] - 1] = temp[i];
           for(int i = 1; i < n; i++) {
        	   brr[i] = Math.min(brr[i], brr[i - 1] + 1);
           }
           for(int i = n - 2; i >= 0; i--) {
        	   brr[i] = Math.min(brr[i], brr[i + 1] + 1);
           }
           for(long e: brr) out.print(e + " ");
           out.println();
             
          }
                 out.close();
			}
		}
