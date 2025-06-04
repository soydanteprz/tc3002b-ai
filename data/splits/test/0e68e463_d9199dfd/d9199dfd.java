import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Scanner;

public class Simple{  
    public static void main(String args[]){  
        //System.out.println("Hello Java");
        Scanner s = new Scanner(System.in);
        int t = s.nextInt();
        while (t>0){
            
            int n = s.nextInt();
            int arr[] = new int[n];

            for(int i=0;i<n;i++){
                arr[i] = s.nextInt();
            }
            String str = s.next();

            //Arrays.sort(arr);
            ArrayList<Integer> blue = new ArrayList<>();
            ArrayList<Integer> red = new ArrayList<>();
            for(int i=0;i<n;i++){
                if(str.charAt(i)=='R'){
                    red.add(arr[i]);
                }
                else{
                    blue.add(arr[i]);
                }
            }
            Collections.sort(red);
            Collections.sort(blue);
            
            int start =1;

            boolean bool =true;
            for(int i=0;i<blue.size();i++){
                if(blue.get(i)<start){
                    bool = false;
                    break;
                }

                start++;
            }
            if(!bool){
                System.out.println("NO");
            }
            else{
                for(int i=0;i<red.size();i++){
                    if(red.get(i)>start){
                        bool = false;
                        break;
                    }
                    start++;
                }
                if(bool){
                    System.out.println("YES");
                }
                else{
                    System.out.println("NO");
                }
            }
            
            


            t--;
        }
        s.close();  
    }
    
}
