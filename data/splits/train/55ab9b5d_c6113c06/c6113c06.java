import java.io.*;
import java.util.*;
public class D
{
    private static int solve(char ch, char[][] str, int N)
    {
        int[] a=new int[N];
        for(int i=0;i<N;i++)
        {
            int c1=0, c2=0;
            for(char x:str[i])
            {
                if(x==ch) c1++;
                else c2++;
            }
            a[i]=c1-c2;
        }

        int sum=0, count=N;
        for(int x:a) sum+=x;

        Arrays.sort(a);
        for(int i=0;i<N;i++)
        {
            if(sum>0) break;
            count--; sum-=a[i];
        }

        return count;
    }
    public static void main(String[] args) throws Exception
    {
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));

        int i,N;

        int T=Integer.parseInt(br.readLine().trim());
        StringBuilder sb=new StringBuilder();

        while (T-->0)
        {
            N=Integer.parseInt(br.readLine().trim());
            char[][] str=new char[N][];

            for(i=0;i<N;i++) str[i]=br.readLine().trim().toCharArray();

            int ans=0;
            for(char ch='a';ch<='e';ch++)
            {
                int tmp=solve(ch,str,N);
                ans=Math.max(ans,tmp);
            }
            sb.append(ans).append("\n");
        }
        System.out.println(sb);
    }
}
