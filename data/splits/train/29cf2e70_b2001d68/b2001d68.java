import java.util.*;
import java.lang.*;
import java.io.*;

public class D753 {
    public static void main(String[] args) throws java.lang.Exception {
        // your code goes here
        try {
            // Scanner sc=new Scanner(System.in);
            FastReader sc = new FastReader();
            int t = sc.nextInt();
            while (t-- > 0) {
                int n=sc.nextInt();
                int[] arr=new int[n];
                for(int i=0;i<n;i++){
                    arr[i]=sc.nextInt();
                }
                
                char[] color=sc.next().toCharArray();
                ArrayList<Integer> b=new ArrayList<>();
                ArrayList<Integer> r=new ArrayList<>();
                for(int i=0;i<n;i++){
                    if(color[i]=='B')b.add(arr[i]);
                    else r.add(arr[i]);
                }
                Collections.sort(b);
                Collections.sort(r);
                boolean ok=true;
                int cur=1;
                for(int i:b){
                    if(i>=cur)cur++;
                    else{
                        ok=false;
                        break;
                    }
                    //cur++;
                }
                for(int i:r){
                    if(i<=cur)cur++;
                    else{
                        ok=false;
                        break;
                    }
                }
                if(ok)System.out.println("YES");
                else System.out.println("NO");
            }
        } catch (Exception e) {
            return;
        }

    }
    

    

    

    

    

    
    

    

    

    

    
}
