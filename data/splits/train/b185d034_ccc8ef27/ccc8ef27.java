import java.util.*;
public class Sol
{
    public static void main(String[] args)
    {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while(t-->0)
        {
			int n = sc.nextInt();
			int a[][]=new int[n][5];
			int tot[]=new int[n];
			for(int i=0;i<n;i++)
			{
				String x = sc.next();
				for(int j=0;j<x.length();j++)
					a[i][x.charAt(j)-'a'] += 1;
				tot[i]=x.length();
			}
			
			int max=Integer.MIN_VALUE;
			for(int i=0;i<5;i++)
				max=Math.max(max,function(a,n,i,tot));
			System.out.println(max);
			
		}	
    }
	static int function(int a[][],int n,int i,int tot[])
	{
		Integer ans[] = new Integer[n];
		for(int j=0;j<n;j++)
			ans[j]=a[j][i]-(tot[j]-a[j][i]);
		int res=0,j=0;
		Arrays.sort(ans,Collections.reverseOrder());
		while(j<n&&res+ans[j]>0)
			res+=ans[j++];
		return j;
	}
}
