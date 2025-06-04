import java.util.Scanner;

public class Main
{	
	public static void main(String[] args)
	{
		Scanner sc=new Scanner(System.in);
		int t=sc.nextInt();
		while(t-->0)
		{
			int n=sc.nextInt();
			int[] c=new int[n];
			for(int i=0;i<n;i++)
				c[i]=sc.nextInt();
			int[] mn= {Integer.MAX_VALUE,Integer.MAX_VALUE};
			long[] rem= {(long)n,(long)n};
			long sum=0;
			long ans=Long.MAX_VALUE;
			for(int i=0;i<n;i++)
			{
				mn[i%2]=Math.min(mn[i%2], c[i]);
				rem[i%2]--;
				sum+=c[i];
				if(i>0)
				{
					long cur=sum+rem[0]*mn[0]+rem[1]*mn[1];
					ans=Math.min(ans, cur);
				}
			}
			System.out.println(ans);
		}
	}
}
