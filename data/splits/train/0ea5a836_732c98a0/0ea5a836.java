import java.util.*;
import java.io.*;
public class Solution
{
    
    //static final long mod=(long)1e9+7;
    static final long mod=998244353L;
    
    
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
        int tc=fs.nextInt();
        while(tc-->0)
        {
            int n=fs.nextInt();
            char[][] c=new char[n][];
            for(int i=0;i<n;i++)
            c[i]=fs.nextLine().toCharArray();
            ArrayList<Integer> list[]=new ArrayList[5];
            for(int i=0;i<5;i++)
            list[i]=new ArrayList<>();
            for(int i=0;i<n;i++)
            {
                int cnt[]=new int[5];
                for(int j=0;j<c[i].length;j++)
                cnt[c[i][j]-'a']++;
                for(int j=0;j<5;j++)
                list[j].add(2*cnt[j]-c[i].length);
            }
            for(int i=0;i<5;i++)
            Collections.sort(list[i]);
            int ans=0;
            for(int i=0;i<5;i++)
            {
                int s=0;
                for(int j=n-1;j>=0;j--)
                {
                    ans=Math.max(ans,n-1-j);
                    s+=list[i].get(j);
                    if(s<=0)
                    break;
                }
                if(s>0)
                ans=n;
            }
            pw.println(ans);
        }
        pw.flush();
        pw.close();
    }
}
