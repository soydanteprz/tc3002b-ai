import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.*;

//import sun.nio.fs.RegistryFileTypeDetector;


 
public class Codeforces {
	static int mod= 998244353;
	
	public static void main(String[] args) throws Exception {
		PrintWriter out=new PrintWriter(System.out);
		FastScanner fs=new FastScanner();
		int t=fs.nextInt();
		
		outer:while(t-->0) {
			int n=fs.nextInt();
			int arr[]=fs.readArray(n);
			int l=0, r=1000000000;
			while(l<r) {
				int mid=(l+r+1)/2;
				if(check(arr,mid)) {
					l=mid;
				}
				else r=mid-1;
			}
			out.println(l);
			
//			System.out.println(check(arr,13));
			
		}
		
		out.close();
		
	}
	static boolean check(int arr[],int min) {
		int n=arr.length;
		int brr[]=new int[n];
		for(int i=n-1;i>1;i--) {
			if(brr[i]+arr[i]<min) return false;
			int cur= Math.min(arr[i], arr[i]+brr[i]-min);
//			System.out.println(cur+" ");
			int d=cur/3;
//			System.out.println(d);
			brr[i-1]+=d;
			brr[i-2]+=2*d;
		}
		
		return (arr[0]+brr[0]>=min)&&(arr[1]+brr[1]>=min);
	}
	
	
	
	
	
	
	
	
	
	// Use this to input code since it is faster than a Scanner
	
 
}
