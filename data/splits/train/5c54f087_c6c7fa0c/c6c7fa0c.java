import java.io.*;
import java.math.BigInteger;
import java.util.*;
import static java.lang.Math.PI;
import static java.lang.System.in;
import static java.lang.System.out;

public class B
{
    static ArrayList<ArrayList<Integer>> adj;
    static int dis[];

    static void dfs(int s, int p, int l)
    {
        dis[s] = l;
        for(int i : adj.get(s))
        {
            if(i==p)    continue;
            dfs(i, s, l+1);
        }
    }

    public static void main(String[] args) throws Exception 
    {
        FastReader sc = new FastReader();
        PrintWriter p = new PrintWriter(out);
        int t = sc.nextInt();
        while(t--!=0)
        {
            int n = sc.nextInt(), a = sc.nextInt(), b = sc.nextInt(), da = sc.nextInt(), db = sc.nextInt();
            adj = new ArrayList<>();
            for(int i = 0; i <= n; i++)
            {
                adj.add(new ArrayList<>());
            }
            for(int i = 1; i < n; i++)
            {
                int u = sc.nextInt(), v = sc.nextInt();
                adj.get(u).add(v);
                adj.get(v).add(u);
            }
            dis = new int[n+1];
            dfs(a, 0, 0);
            if(dis[b] <= da)    //Alice reached in first move
            {
                p.println("Alice");
                continue;
            }
            int farthest = 0, maxDis = 0;
            for(int i = 1; i <= n; i++)
            {
                if(maxDis < dis[i])
                {
                    maxDis = dis[i];
                    farthest = i;
                }
            }
            dfs(farthest, 0, 0);
            int diameter = 0;
            for(int i = 1; i <= n; i++)
            {
                diameter = Math.max(diameter, dis[i]);
            }
            if(db > 2*da && diameter > 2*da)
            {
                p.println("Bob");
            }
            else
            {
                p.println("Alice");
            }
        }
        p.close();
    }
/* 
*/    
/*
1. Check overflow in pow function or in general
2. Check indices of read array function
3. Think of an easier solution because the problems you solve are always easy
4. Check iterator of loop
5. If still doesn't work, then jump from the 729th floor 'coz "beta tumse na ho paayega"

    Move to top!!
*/
}
