import java.io.*;
import java.util.*;


public class CP
{
	static Scanner sc=new Scanner(System.in);
	public static void main(String[] args) throws IOException, CloneNotSupportedException
	{
		int t=sc.nextInt();
		while(t-->0)
		{
			int n=sc.nextInt();
			int k[]=new int[n];
			int h[]=new int[n];
			for(int i=0;i<n;i++)
				k[i]=sc.nextInt();
			for(int i=0;i<n;i++)
				h[i]=sc.nextInt();
			ArrayList<Pair> interval=new ArrayList<Pair>();
			ArrayList<Pair> act=new ArrayList<Pair>();
			for(int i=0;i<n;i++)
				interval.add(new Pair(k[i]-h[i]+1,k[i]));
			Collections.sort(interval);
//			out.println(interval);
			act.add(interval.get(0).clone());
			for(int i=1;i<n;i++)
			{
				Pair p=act.get(act.size()-1);
				if(p.y<interval.get(i).x)
					act.add(interval.get(i).clone());
				else
					p.y=Math.max(p.y, interval.get(i).y);
			}
//			out.println(act);
			long mana=0;
			for(Pair p: act)
			{
				long x=p.y-p.x+1;
				mana+=(x*(x+1))/2;
			}
			System.out.println(mana);		
 		}
	}
	static class Pair implements Cloneable, Comparable<Pair>
	{
		int x,y;
		Pair(int a,int b)
		{
			this.x=a;
			this.y=b;
		}
//		@Override
//		public boolean equals(Object obj) 
//		{
//			if(obj instanceof  Pair)
//			{
//				Pair p=(Pair)obj;
//				return p.x==this.x && p.y==this.y;
//			}
//			return false;
//		}
//		@Override
//		public int hashCode() 
//		{
//			return Math.abs(x)+500*Math.abs(y);
//		}
//		@Override
//		public String toString() 
//		{
//			return "("+x+" "+y+")";
//		}
		@Override
		protected Pair clone() throws CloneNotSupportedException {
			return new Pair(this.x,this.y);
		}
		
//		public void swap()
//		{
//			this.y=this.y+this.x;
//			this.x=this.y-this.x;
//			this.y=this.y-this.x;			
//		}
	}
}
