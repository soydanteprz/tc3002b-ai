import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
 
public class a729 {
 
	public static void main(String[] args) throws IOException {
   
		
//		try {
			
		BufferedWriter out = new BufferedWriter(
	            new OutputStreamWriter(System.out));
		BufferedReader br = new BufferedReader(
		        new InputStreamReader(System.in));
		PrintWriter pt = new PrintWriter(System.out);
		FastReader sc = new FastReader();
		int t = sc.nextInt();
	
for(int o = 0 ; o<t;o++){
int n = sc.nextInt();
long k = sc.nextLong();
long[] arr = new long[n];
for(int i = 0 ; i<n;i++) {
	arr[i] = sc.nextLong();
}
shuffleArray(arr);
Arrays.sort(arr);
long[] psum = new long[n+1];
for(int i = 1; i<=n;i++){
	psum[i] = psum[i-1] + arr[i-1];
}
if(n==1) {
	long val = arr[0]-k;
	val = Math.max(0, val);
	System.out.println(val);
	continue;
}
if(k>=psum[n]) {
	System.out.println(0);
	continue;
}
long ans = Integer.MAX_VALUE;
long x = 0;
long a1 = arr[0];
for(int y = 0 ; y<n;y++) {
	long val = psum[n-y] + y*a1 - k;
//	if(val<0) {
//		continue;
//	}
	x = val/(y+1);
	if(val%(y+1)!=0 && val>0) {
		x++;
	}
//	if(x<0) {
//		continue;
//	}
	ans = Math.min(ans, Math.max(x, 0) + y);
	
	
}
ans = Math.max(ans, 0);
System.out.println(ans);
}
//     }catch(Exception e) {
//    	 return;
//     }
     }
     
     
//------------------------------------------------------------------------------------------------------------------------------------------------
 
	
	
	
	
	
	
	
	
	
 
	
	
	
	
//for f/inding minimum in range 

//// for sum
//

	
	
	
	
	
	
	
	
	
 
//	-----------------------------------------------------------------------------------------------------------------------------------------------
	
//--------------------------------------------------------------------------------------------------------------------------------------------------------
	
 
		
 
//-----------------------------------------------------------------------------------------------------------------------------------
 
	//segment tree
	//for finding minimum in range 
//	public static void build(int [] seg,int []arr,int idx, int lo , int hi) {
//		if(lo == hi) {
//			seg[idx] = arr[lo];
//			return;
//		}
//		int mid = (lo + hi)/2;
//	build(seg,arr,2*idx+1, lo, mid);
//	build(seg,arr,idx*2+2, mid +1, hi);
//	seg[idx] = Math.min(seg[2*idx+1],seg[2*idx+2]);
//	}
////for finding minimum in range 
//public static int query(int[]seg,int idx , int lo , int hi , int l , int r) {
//		if(lo>=l && hi<=r) {
//			return seg[idx];
//		}
//	if(hi<l || lo>r) {
//		return Integer.MAX_VALUE;
//	}
//	int mid = (lo + hi)/2;
//	int left = query(seg,idx*2 +1, lo, mid, l, r);
//	int right = query(seg,idx*2 + 2, mid + 1, hi, l, r);
//	return Math.min(left, right);
//}
//	// for sum
//
//public static void update(int[]seg,int idx, int lo , int hi , int node , int val) {
//	if(lo == hi) {
//		seg[idx] += val;
//	}else {
//int mid = (lo + hi )/2;
//if(node<=mid && node>=lo) {
//	update(seg, idx * 2 +1, lo, mid, node, val);
//}else {
//	update(seg, idx*2 + 2, mid + 1, hi, node, val);
//}
//seg[idx] = seg[idx*2 + 1] + seg[idx*2 + 2];
//
//}
 
//}
//---------------------------------------------------------------------------------------------------------------------------------------
 
 
	static void shuffleArray(long[] ar)
	  {
	    // If running on Java 6 or older, use `new Random()` on RHS here
	    Random rnd = ThreadLocalRandom.current();
	    for (int i = ar.length - 1; i > 0; i--)
	    {
	      int index = rnd.nextInt(i + 1);
	      // Simple swap
	      long a = ar[index];
	      ar[index] = ar[i];
	      ar[i] = a;
	    }
	  }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------
 
}
 
 

//------------------------------------------------------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------------------------------------------------

 

