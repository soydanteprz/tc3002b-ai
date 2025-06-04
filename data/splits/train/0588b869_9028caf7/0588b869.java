import java.util.*;
import java.io.*;
public class Solution
{
    
    static final long mod=(long)1e9+7;
    
    
    /*static class Pair implements Comparable<Pair>
    {
        int v,l;
        Pair(int v,int l)
        {
            this.v=v;
            this.l=l;
        }
        public int compareTo(Pair p)
        {
            return l-p.l; 
        }
    }*/
    
    
    public static void main(String args[])throws Exception
    {
        FastReader fs=new FastReader();
        PrintWriter pw=new PrintWriter(System.out);
        //int tc=fs.nextInt();
        int n=fs.nextInt();
        int a[]=new int[n];
        for(int i=0;i<n;i++)
        a[i]=fs.nextInt();
        ArrayList<Integer> o=new ArrayList<>();
        ArrayList<Integer> z=new ArrayList<>();
        for(int i=0;i<n;i++)
        {
            if(a[i]==1)o.add(i);
            else z.add(i);
        }
        int ans[][]=new int[o.size()+1][z.size()+1];
        for(int i=1;i<=o.size();i++)
        {
            for(int j=i;j<=z.size();j++)
            {
                if(i==j)ans[i][j]=ans[i-1][j-1]+(int)Math.abs(o.get(i-1)-z.get(j-1));
                else 
                ans[i][j]=Math.min(ans[i][j-1],ans[i-1][j-1]+(int)Math.abs(o.get(i-1)-z.get(j-1)));
            }
        }
        pw.println(ans[o.size()][z.size()]);
        pw.flush();
        pw.close();
    }
}
