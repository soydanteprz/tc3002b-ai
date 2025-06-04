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
	String s1 = sc.next();
	String s2 = sc.next();
	if(s1.equals(s2)) {
		System.out.println(0);
		continue;
	}
	int f = 0;
	for(int i = 0 ; i<n;i++) {
		if(s1.charAt(i) == '1') {
			f = 1;
			break;
		}
	}
if(f == 0) {
	System.out.println(-1);
continue;
}
int c = 0;
int c1 = 0;
int c2 = 0;
int c3 = 0;
int c4 = 0;
for(int i = 0; i<n;i++) {
	if(s1.charAt(i)!=s2.charAt(i)) {
		c++;
	if(s1.charAt(i) == '1') {
		c1++;
	}else {
		c2++;
	}
	}else {
		if(s1.charAt(i) == '1') {
			c3++;
		}else {
			c4++;
		}
	}
}

int g = 0;
int h = 0;

int v1 = Integer.MAX_VALUE;
int v2 = Integer.MAX_VALUE;
if(c%2==1) {
g = 1;
}

int e = n-c;
if(e%2==0) {
h = 1;
}
//System.out.println(Math.min(v1, v2));
if(c2!=c/2){
	g = 1;
}
if(c4!=e/2){
	h = 1;
}
if(g == 0){
	v1 = c;
}
if(h == 0){
	v2 = e;
}

int ans = Math.min(v1, v2);

if(ans == Integer.MAX_VALUE) {
	System.out.println(-1);
}else {
	System.out.println(ans);
}
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


	
//-----------------------------------------------------------------------------------------------------------------------------------------------------------

}



//------------------------------------------------------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------------------------------------------------



