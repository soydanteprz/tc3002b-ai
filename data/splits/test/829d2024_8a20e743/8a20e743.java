import java.io.*;
import java.util.*;
public class C
{ 
	static class Pair{
		int h,in;
		Pair(){}
		Pair(int in,int h){
			this.in=in;
			this.h=h;
		}
	}
	static class customSort implements Comparator<Pair>{
		
	}
    public static void main(String[] args) 
	{ 
		FastReader sc=new FastReader(); 
		StringBuffer ans=new StringBuffer();
		int test=sc.nextInt();
		while(test-->0)
		{
			
			int n=sc.nextInt(),m=sc.nextInt(),x=sc.nextInt();
			
			int arr[]=new int[n];
			int res[]=new int[n];
			
			int sum =0;
			for(int i=0;i<n;i++) {
				arr[i]=sc.nextInt();sum+=arr[i];
			}
			
			PriorityQueue<Pair> pq=new PriorityQueue<>(new customSort());
			
			for(int i=1;i<=m;i++) {
				pq.add(new Pair(i,0));
			}
			
			ans.append("YES\n");
			for(int a:arr) {
				Pair xx=pq.poll();
				ans.append(xx.in+" ");
				xx.h+=a;
				pq.add(xx);
			}
			ans.append("\n");
			
			
		}
	    System.out.print(ans);
	} 
    
 
    
    
    static final Random random=new Random();
    
	
	
	
	
	 


}
