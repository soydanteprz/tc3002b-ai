import java.util.*;
import java.io.*;
public class CodeForces {
	// For fast input output
	

	// end of fast i/o code
	public static void main(String[] args) {
		FastReader in = new FastReader();
		OutputStream op = System.out;
		PrintWriter out = new PrintWriter(op);
		int t = in.nextInt();
		for (int i = 1; i <= t; i++) {
			int n=in.nextInt();
			int arr[]=new int[n];
			for(int j = 0; j < n; j++)
				arr[j]=in.nextInt();
			helper(n,arr,out);
			out.println();
		}
		out.close();

	}

public static void helper(int n,int arr[],PrintWriter o) 
	{
		int max=Integer.MIN_VALUE;
		for(int i=2;i<n;i++)
				max=Math.max(max,arr[i]);
		int ans=1,low=1,high=max;
		while(low<=high)
		{
			int mid=low+(high-low)/2;
			if(canFit(mid,arr))
				{
					ans=mid;
					low=mid+1;
				}
				else 
					high=mid-1;
		}
		o.print(ans);
	}
	static	boolean canFit(int mid,int arr[])
	{
		int copy[]=Arrays.copyOf(arr, arr.length);
		for(int i=arr.length-1;i>=0;i--)
		{
			if(copy[i]<mid)
				return false;
			int min=Math.min(copy[i]-mid,arr[i])/3;
			if(i>=2){
			copy[i-1]+=min;
			copy[i-2]+=2*min;
		}}
		return true;
	}
}
