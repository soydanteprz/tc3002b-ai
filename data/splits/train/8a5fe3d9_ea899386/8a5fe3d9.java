import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;

public class Main {
    static class pair implements Comparable<pair>{
        int sum, index;
        pair(int i){
            sum = 0;
            index = i;
        }
        
    }
    public static void main(String[] args) throws IOException{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();
        int t = Integer.parseInt(br.readLine());
        while(t-- > 0){
            StringTokenizer st = new StringTokenizer(br.readLine());
            int n = Integer.parseInt(st.nextToken());
            int m = Integer.parseInt(st.nextToken());
            int x = Integer.parseInt(st.nextToken());
            int ar[] = new int[n];
            st = new StringTokenizer(br.readLine());
            int sum = 0, low = x;
            for(int i = 0; i < n; i++){
                ar[i] = Integer.parseInt(st.nextToken());
            }
            PriorityQueue<pair> q = new PriorityQueue<pair>();
            for(int i = 0; i < m; i++)
                q.add(new pair(i + 1));
            sb.append("YES\n");
            for(int i = 0; i < n; i++){
                pair p = q.poll();
                p.sum += ar[i];
                q.add(p);
                sb.append(p.index + " ");
            }
            sb.append("\n");
        }
        System.out.print(sb);
    }
}
