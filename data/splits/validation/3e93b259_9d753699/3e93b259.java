import java.util.*;
import java.io.*;
public class Main
{
    
    
    public static void main(String args[])
    {
        FastReader fs=new FastReader();
        PrintWriter pw=new PrintWriter(System.out);
        int tc=fs.nextInt();
        while(tc-->0)
        {
            int n=fs.nextInt();
            long a[]=new long[n];
            for(int i=0;i<n;i++)
            a[i]=fs.nextLong();
            long minans=a[0]*n+a[1]*n;
            long sum=a[0]+a[1];
            long min1=a[0],min2=a[1];
            for(int i=2;i<n;i++)
            {
                sum+=a[i];
                if(i%2==0)
                min1=Math.min(min1,a[i]);
                else
                min2=Math.min(min2,a[i]);
                long tsum=sum-min1-min2;
                if(i%2==0)
                tsum=tsum+(n-i/2+1)*min2+(n-i/2)*min1;
                else
                tsum=tsum+(min1+min2)*(n-((i+1)/2)+1);
                
                minans=Math.min(minans,tsum);
                //minans=Math.min(minans,(n-cnt1)*a[i]+(n-cnt2)*a[i-1]+sum);
            }
            pw.println(minans);
        }
        pw.flush();
        pw.close();
    }
}
