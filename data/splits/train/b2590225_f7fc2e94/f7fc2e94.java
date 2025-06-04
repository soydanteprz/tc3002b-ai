import java.io.*;
import java.util.*;
public class codefor 
{
	static class height implements Comparable<height>
	{
		int h,index;
		height(int hi,int i)
		{
			h=hi;
			index=i;
		}
		
		
	}
	 
	public static void main(String[] args) 
	{
		FastReader sc=new FastReader();
		int T=sc.nextInt();
		while(T-->0)
		{
			int n=sc.nextInt(),m=sc.nextInt(),x=sc.nextInt(),i;
			ArrayList<height> arr=new ArrayList<>();
			int ans[]=new int[n];
			PriorityQueue<height> tower=new PriorityQueue<height>();
			for(i=0;i<n;i++)
				arr.add(new height(sc.nextInt(),i));
			if(n<m)
				System.out.println("NO");
			else
			{
				System.out.println("YES");
			Collections.sort(arr,Collections.reverseOrder());
			for(i=0;i<m;i++)
				tower.add(new height(0,i+1));
			for(i=0;i<n;i++)
			{
				//System.out.print(tower);
				height t=tower.poll();
				t.h=t.h+arr.get(i).h;
				//System.out.println(" "+t);
				tower.add(t);
				ans[arr.get(i).index]=t.index;
			}
			for(i=0;i<n;i++)
				System.out.print(ans[i]+" ");
			System.out.println();
			}
		}
	}
}
