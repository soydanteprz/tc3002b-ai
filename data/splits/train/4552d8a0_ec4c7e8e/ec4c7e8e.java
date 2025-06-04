import java.util.*;
public class the_child_and_set
{
	public static void main(String args[])
	{
		Scanner in=new Scanner(System.in);
		int t=in.nextInt();
		in.nextLine();
		while(t--!=0)
		{
			int n=in.nextInt();
			int arr[]=new int[n];
			for(int i=0;i<n;i++)
			{
				arr[i]=in.nextInt();
			}
			in.nextLine();
			String s=in.nextLine();
			int rc=0;
			int bc=0;
			for(int i=0;i<n;i++)
			{
				if(s.charAt(i)=='B')
					bc++;
				else
					rc++;
			}
			int r[]=new int[rc];
			int b[]=new int[bc];
			int bi=0;
			int ri=0;
			for(int i=0;i<n;i++)
			{
				if(s.charAt(i)=='B')
				{
					b[bi]=arr[i];
					bi++;
				}
				else
				{
					r[ri]=arr[i];
					ri++;
				}
			}
			Arrays.sort(b);
			Arrays.sort(r);
			boolean flag=true;
			for(int i=0;i<bi;i++)
			{
				if(b[i]<(i+1))
				{
					flag=false;
					break;
				}
			}
			if(flag)
			{
				for(int i=0;i<ri;i++)
				{
					if(r[i]>bi+(i+1))
					{
						flag=false;
						break;
					}
				}
			}
			if(flag)
				System.out.println("YES");
			else
				System.out.println("NO");
		}
	}
}
