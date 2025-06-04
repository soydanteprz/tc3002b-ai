import java.util.*;
import java.io.*;

public class Solution {

    static class Pair {
        int x;
        int y;
        Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }

        

        
    }

    public static void main(String[] args) throws IOException {

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        int t = Integer.parseInt(br.readLine());

        BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));

        StringTokenizer st;

        while (t-- > 0) {

            int n = Integer.parseInt(br.readLine());
            ArrayList<ArrayList<Integer>> adj = new ArrayList<>();

            for (int i = 0; i <= n; i++)
                adj.add(new ArrayList<>());

            ArrayList<Pair> list2 = new ArrayList<>();
            HashMap<Pair, Integer> map = new HashMap<>();

            for (int i = 0; i < n - 1; i++) {
                st = new StringTokenizer(br.readLine());
                int k1 = Integer.parseInt(st.nextToken());
                int k2 = Integer.parseInt(st.nextToken());
                if (k1 > k2) {
                    int temp4 = k2;
                    k2 = k1;
                    k1 = temp4;
                }
                Pair temp = new Pair(k1, k2);
                list2.add(temp);
                adj.get(k1).add(k2);
                adj.get(k2).add(k1);
            }

            int endp = -1;
            for(int i = 1; i <=n; i++)
            {
                if(adj.get(i).size() == 1)
                {
                    endp = i;
                    break;
                }
            }

            int flag = 0;
            if(endp == -1)
                flag = 1;
            Queue<Integer> q = new LinkedList<>();
            q.add(endp);
            int a = 2;
            int vis[] = new int[n+1];
            vis[endp] = 1;
            while(!q.isEmpty())
            {
                int node = q.poll();
                ArrayList<Integer> temp = adj.get(node);
                if(temp.size() > 2)
                {
                    flag = 1;
                    break;
                }
                for(int i = 0; i < temp.size(); i++)
                {
                    if(vis[temp.get(i)] == 0)
                    {
                        vis[temp.get(i)] = 1;
                        q.add(temp.get(i));
                        int res = 7 - a;
                        a = res;
                        Pair p = new Pair(Math.min(node, temp.get(i)), Math.max(temp.get(i), node));
                        map.put(p, res);
                    }
                }
            }

            if(flag == 1)
            {
                output.write("-1\n");
                continue;
            }
            for(Pair p : list2)
                output.write(map.get(p) + " ");

            output.write("\n");
        }

        output.flush();

    }

}
