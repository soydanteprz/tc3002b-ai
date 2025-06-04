import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;



public class D_1552 {
	 
    public static void main(String args[])
    {
    	 FastReader fs = new FastReader();
        int tc = fs.nextInt();
        
        while(tc-- > 0)
        {   boolean flag=false;
        	int n=fs.nextInt();
            int[] ar=new int[n];
            for(int i=0;i<n;i++)
            {ar[i]=fs.nextInt();}
        	for(int i=1;i<Math.pow(3, n);i++)
        	{
        		int copy=i;
        		int sum=0;
        		for(int j=0;j<n;j++)
        		{
        			int rem=copy%3;
        			sum+=rem==0?0:rem==1?ar[j]:-ar[j];
        			copy=copy/3;
        		}
        		if(sum==0)
        			{System.out.println("yes");
        			flag=true;
        			break;}
        			}
        	       if(flag==false)
        			System.out.println("no");
        	}
        
        }
        
    
    
}
