import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;
 
 
public class Codeforces {
	
	static int mod =1000000007; 
	static Set<Integer> set;
	public static void main(String[] args) throws Exception {
		PrintWriter out=new PrintWriter(System.out);
	    FastScanner fs=new FastScanner();
	    int t=fs.nextInt();
	    while(t-->0) {
	    	int n=fs.nextInt();
	    	int arr[]=new int[n];
//	    	set=new HashSet<>();
	    	for(int i=0;i<n;i++) {
	    		arr[i]=Math.abs(fs.nextInt());
//	    		set.add(arr[i]);
	    	}
//	    	sort(arr);
//	    	for(int i=0;i<n;i++) System.out.print(arr[i]+" ");
//	    	System.out.println();
//	    	if(set.size()<n||set.contains(0)) {
//	    		System.out.println("YES");
//	    		continue;
//	    	}
	    	boolean f=false;
	    	for(int i=0;i<n;i++) {
	    		boolean cur=recur(0,i,arr,0);
	    		if(cur) {
	    			f=true;
	    			break;
	    		}
	    	}
	    	if(f) System.out.println("YES");
	    	else System.out.println("NO");
	    }
	    out.close();
	}
	static boolean recur(int pos,int ind,int arr[],int sum) {
		if(pos==ind) return recur(pos+1,ind,arr,sum);
		if(sum==arr[ind]) return true;
		if(pos==arr.length) {
			return false;
		}
		
		if(recur(pos+1,ind,arr,sum+arr[pos])) return true;
		if(recur(pos+1,ind,arr,sum)) return true;
		if(recur(pos+1,ind,arr,sum-arr[pos])) return true;
		return false;
	}
	
	
	
	
	
	
	
	
	// Use this to input code since it is faster than a Scanner
	
 
}
