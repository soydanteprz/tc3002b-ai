import java.util.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.StringTokenizer;

//import graphs.Segment_Trees.FastReader;

public class PW {
	
	//private static BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));
	
	public static int memo[][];
	public static FastReader s = new FastReader();
	public static List<Long> primes;
	public static int sz= 1000000;
	//public static int arr[];
	public static int MOD=1000000007;
	public static List<List<Integer>> adj;
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	 
		//FastScanner sc=new FastScanner();
		 
		int t=s.nextInt();
		
		while(t-- >0)
		{
			
			//int n=0,a=0,b=0;
			int n=s.nextInt();
			int k=s.nextInt();
	
			  long arr[]=new long[n];
			  Arrays.fill(arr,Long.MAX_VALUE/10);
			  int ac[]=new int[k];
			  long tmp[]=new long[k];
			  for(int i = 0 ; i < k ; i++){
			    ac[i]=s.nextInt();
			    ac[i]--;
			  }
			  for(int i = 0 ; i < k ; i++){
			    tmp[i]=s.nextLong();
			    arr[ac[i]] = tmp[i];
			  }
			  long P[] = getP(arr, n);
			  
			  int i=0,j=arr.length-1;
			  while(i<=j)
			  {
				  long temp=arr[i];
				  arr[i]=arr[j];
				  arr[j]=temp;
				  i++;
				  j--;
			  }
			  //reverse(all(arr));
			  long[] S = getP(arr, n);
			  //reverse(all(S));
			  i=0;j=S.length-1;
			  while(i<=j)
			  {
				  long temp=S[i];
				  S[i]=S[j];
				  S[j]=temp;
				  i++;
				  j--;
			  }
			  for(i = 0 ; i < n ; i++){
			    System.out.print(Math.min(P[i], S[i])+" ");
			  }
			  System.out.println();
			
		}
		
	}
	
	public static long[] getP(long []arr, int n){
		  long mn = Long.MAX_VALUE/10;
		  long P[]=new long[n];
		  Arrays.fill(P, Long.MAX_VALUE/10);
		  int cnt = 0;
		  for(int i = 0 ; i < n ; i++){
		    long curr = arr[i];
		    if(mn + cnt < curr){
		      P[i] = mn + cnt;
		    }else{
		      mn = arr[i];
		      P[i] = arr[i];
		      cnt  = 0;
		    }
		    cnt++;
		  }
		  return  P;
		}
		
	
		
	
	
	
	
	
	

	
	
	
	
	
//	public static long fastExpo(long a, long n, long mod) {
//		  long result = 1;
//		  while (n > 0) {
//		    if ((n & 1)>0)
//		      result = (result * a) % mod;
//		    a = (a * a) % mod;
//		    n >>= 1;
//		  }
//		  return result;
//		}
//	1  4   8
	
	
	 
	
	 
	
	
	
	
	
	
	
	
	    public static String largestNumber( List<String> ab) {
	        
	       // List<String> ab= new ArrayList<>();
	       // for(int i=0;i<A.size();i++)
	       // {
	       //     ab.add(String.valueOf(A.get(i)));
	       // }
	        
	        Collections.sort(ab, new Comparator<String>(){ 
	            
	            
	            
	            
	            
	        });
	        StringBuilder abc= new StringBuilder();
	        
	        
	        for(int i=0;i<ab.size();i++)
	        {
	           abc.append(ab.get(i));
	            
	        }
	        if(abc.length()==0)
	            return abc.toString();
	        
	        if(abc.charAt(0)=='0')
	        return "0";
	        else
	        return abc.toString();
	        
	    }
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	
	
	
}

 
