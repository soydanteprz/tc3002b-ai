import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
 
public class C {
 
	public static void main(String[] args) {
		FastScanner fs=new FastScanner();
		int t=fs.nextInt();
		while(t-->0) 
		{
			int n;
			long k,sum=0L,temp=0L,min=Integer.MAX_VALUE;
			n=fs.nextInt();
			k=fs.nextLong();
			int[] arr=fs.readArray(n);
		    for(int i=0;i<n;i++)
		        sum+=arr[i];
		    sort(arr);
		    Stack<Integer> s=new Stack<>();
		    for(int i=1;i<n;i++){
		        s.push(arr[i]-arr[0]);
		    }
		    while(!s.empty() && sum>k){
		        min=Math.min(min,temp+(sum-k+temp)/(temp+1));
		        sum-=s.peek();
		        s.pop();
		        temp++;
		    }
		    if(sum>k){
		    	temp+=(sum-k+temp)/(temp+1);
		    }
		    System.out.println(Math.min(min, temp));
		}
	}
	
	
 
	static void sort(int[] a) {
		ArrayList<Integer> l=new ArrayList<>();
		for (int i:a) l.add(i);
		Collections.sort(l);
		for (int i=0; i<a.length; i++) a[i]=l.get(i);
	}
	
	
	
 
	
}
