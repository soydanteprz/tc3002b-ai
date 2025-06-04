import java.io.*;
import java.lang.*;
import java.util.*;

public class E1578 {
    public static void main(String[] args) throws IOException{
        StringBuffer ans = new StringBuffer();
        StringTokenizer st;
        BufferedReader f = new BufferedReader(new InputStreamReader(System.in));
        st = new StringTokenizer(f.readLine());
        int q = Integer.parseInt(st.nextToken());
        for(; q > 0; q--){
            st = new StringTokenizer(f.readLine());
            st = new StringTokenizer(f.readLine());
            int n = Integer.parseInt(st.nextToken());
            int k = Integer.parseInt(st.nextToken());
            int[] a = new int[k];
            st = new StringTokenizer(f.readLine());
            for(int i = 0; i < k; i++){
                a[i] = Integer.parseInt(st.nextToken()) - 1;
            }
            int[] arr = new int[n];
            int[] b = new int[k];
            st = new StringTokenizer(f.readLine());
            Arrays.fill(arr, Integer.MAX_VALUE-1);
            for(int i = 0; i < k; i++){
                b[i] = Integer.parseInt(st.nextToken());
                arr[a[i]] = b[i];
            }
            int[] l = new int[n];
            int[] r = new int[n];
            for(int i = 0; i < n; i++){
                if(i == 0)
                    l[i] = arr[i];
                else
                    l[i] = Math.min(l[i-1]+1, arr[i]);
            }
            for(int i = n-1; i > -1; i--){
                if(i == n-1)
                    r[i] = arr[i];
                else
                    r[i] = Math.min(r[i+1]+1, arr[i]);
            }
            for(int i = 0; i  < n; i++)
                ans.append(Math.min(l[i], r[i])).append(" ");
            ans.append("\n");
        }

        f.close();



        System.out.println(ans);
    }
    

}
