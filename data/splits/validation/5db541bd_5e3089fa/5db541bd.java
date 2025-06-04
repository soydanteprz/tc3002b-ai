import java.io.*;

import java.util.*;
/*




*/

 
 public class C{
	static FastReader sc=null;
	
	public static void main(String[] args) {
		sc=new FastReader();
		int t=sc.nextInt();
		
		for(int tt=0;tt<t;tt++) {
			int n=sc.nextInt();
			long k=sc.nextLong();
			int a[]=sc.readArray(n);
			ruffleSort(a);
			
			long sum=0;
			for(int e:a)sum+=e;
			
			if(sum<=k) {
				System.out.println(0);
				continue;
			}
			long l=-1,r=sum-k+1;
			while(l+1<r) {
				long mid=(l+r)/2;
				if(pos(mid,k,a,sum))r=mid;
				else l=mid;
			}
			
			System.out.println(r);
		}
	}
	
	static boolean pos(long m,long k,int a[],long sum) {
		int n=a.length;
		
		//we set the largest of them to be equal val
		long min=sum-m,suf=sum-a[0],moves=m;
		
		for(int i=n-1;i>0;i--) {
			moves--;
			if(moves<0)break;
			long val=a[0]-moves;
			suf-=a[i];
			long tot=suf+val*(n-i+1);
			min=Math.min(min, tot);
		}
		
		return min<=k;
	}
	
	static int[] ruffleSort(int a[]) {
		ArrayList<Integer> al=new ArrayList<>();
		for(int i:a)al.add(i);
		Collections.sort(al);
		for(int i=0;i<a.length;i++)a[i]=al.get(i);
		return a;
	}
	
	
	
	
	
	
}
