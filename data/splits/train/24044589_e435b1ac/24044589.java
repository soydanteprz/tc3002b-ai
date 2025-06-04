import java.util.*;
import java.io.*;

public class C_Phoenix_and_Towers{
    public static void main(String[] args) {
        FastScanner s= new FastScanner();
        StringBuilder res = new StringBuilder();
        int t=s.nextInt();
        int p=0;
        while(p<t){
        int n=s.nextInt();
        int m=s.nextInt();
        int x=s.nextInt();
        int array[]= new int[n];
        HashMap<Integer,Queue<Integer>> indices = new HashMap<Integer,Queue<Integer>>();
        for(int i=0;i<n;i++){
            array[i]=s.nextInt();
            if(indices.containsKey(array[i])){
              indices.get(array[i]).add(i);
            }
            else{
                Queue<Integer> obj = new LinkedList<Integer>();
                obj.add(i);
                indices.put(array[i],obj);
            }
        }
        long tower[]= new long[m];
        int ans[]= new int[n];
        Arrays.sort(array);
        int k=0;
        for(int i=0;i<n;i++){
            tower[k]+=array[i];
            int a=indices.get(array[i]).poll();
            ans[a]=k+1;
            k++;
            if(k==m){
                k=0;
            }

        }
        long max=Long.MIN_VALUE;
        long min=Long.MAX_VALUE;
        for(int i=0;i<m;i++){
            min=Math.min(min,tower[i]);
            max=Math.max(max,tower[i]);
        }
        if(max-min<=x){
            res.append("YES\n");
            for(int i=0;i<n;i++){
                res.append(ans[i]+" ");
            }
            res.append("\n");
        }
        else{
            res.append("NO\n");
        }
        p++;
    }
    System.out.println(res);

    }
    




}
