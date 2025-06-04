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
			int k= sc.nextInt();
			int ac[] = new int[k];
			int temp[] = new int[k];
			int ans[] = new int[n+1];
			for(int i=0;i<k;i++)
				ac[i] = sc.nextInt();
			for(int i=0;i<k;i++)
				temp[i] = sc.nextInt();
			Arrays.fill(ans,Integer.MAX_VALUE);
			for(int i=0;i<k;i++)
				ans[ac[i]] = temp[i];

			for(int i=2;i<n+1;i++)
				ans[i]=Math.min(ans[i],ans[i-1]==Integer.MAX_VALUE ? ans[i-1] : ans[i-1]+1);

			for(int i=n-1;i>0;i--)
				ans[i]=Math.min(ans[i],ans[i+1]+1);
			
			for(int i=1;i<n+1;i++)
				System.out.print(ans[i]+" ");
		
			System.out.println(" ");
		}	
    }
}
